import torch
import torch.nn as nn
import jieba
import string
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import time

from .TranslateAPI import translate
import nltk
nltk.download('averaged_perceptron_tagger')

torch.autograd.set_detect_anomaly(True)


class BaseAttack:
    def __init__(self, model, tokenizer, device, config, space_token):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model = self.model.to(self.device)

        self.embedding = self.model.get_input_embeddings().weight
        self.specical_token = self.tokenizer.all_special_tokens
        self.specical_id = self.tokenizer.all_special_ids
        self.eos_token_id = self.model.config.eos_token_id
        self.pad_token_id = self.model.config.pad_token_id
        self.space_token = space_token

        self.num_beams = config['num_beams']
        self.num_beam_groups = config['num_beam_groups']
        self.max_per = config['max_per']
        self.max_len = config['max_len']
        self.source_language = config['src']
        self.target_language = config['tgt']

        self.softmax = nn.Softmax(dim=1)
        self.bce_loss = nn.BCELoss()

    def run_attack(self, x):
        pass

    def compute_loss(self, x):
        pass

    def compute_seq_len(self, seq):
        if seq[0].eq(self.pad_token_id):
            return int(len(seq) - sum(seq.eq(self.pad_token_id)))
        else:
            return int(len(seq) - sum(seq.eq(self.pad_token_id))) - 1

    def get_prediction(self, text):
        def remove_pad(s):
            for i, tk in enumerate(s):
                if tk == self.eos_token_id and i != 0:
                    return s[:i + 1]
            return s

        input_token = self.tokenizer(text, return_tensors="pt", padding=True).input_ids
        input_token = input_token.to(self.device)

        out_token = translate(
            self.model, input_token,
            early_stopping=False, num_beams=self.num_beams,
            num_beam_groups=self.num_beam_groups, use_cache=True,
            max_length=self.max_len
        )

        # 原始输出
        all_seqs = out_token['sequences']
        all_scores = out_token['scores']

        # 每个样本保留第一个 beam 的输出和得分
        num_samples = len(text)
        seqs = []
        out_scores = []

        for i in range(num_samples):
            beam_start = i * self.num_beams
            beam_seq = remove_pad(all_seqs[beam_start])
            beam_scores = [score[beam_start].detach().clone().requires_grad_(True) for score in all_scores]
            seqs.append(beam_seq)
            out_scores.append(torch.stack(beam_scores))  # shape: [seq_len, vocab_size]

        pred_len = [self.compute_seq_len(seq) for seq in seqs]
        return pred_len, seqs, out_scores




    def get_trans_string_len(self, text):
        pred_len, seqs, _ = self.get_prediction(text)
        return seqs[0], pred_len[0]

    def get_trans_len(self, text):
        pred_len, _, _ = self.get_prediction(text)
        return pred_len

    def get_trans_strings(self, text):
        pred_len, seqs, _ = self.get_prediction(text)
        out_res = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in seqs ]
        return out_res, pred_len

    def compute_score(self, text): 
        batch_size = len(text)
        index_list = [i * self.num_beams for i in range(batch_size + 1)]

        pred_len, seqs, out_scores = self.get_prediction(text)

        scores = [[] for _ in range(batch_size)]
        for out_s in out_scores:
            for i in range(batch_size):
                current_index = index_list[i]
                scores[i].append(out_s[current_index: current_index + 1])
        scores = [torch.cat(s) for s in scores]

        # ✅ 安全切片（避免 inplace 问题）
        scores = [s.clone()[:pred_len[i]] for i, s in enumerate(scores)]
        return scores, seqs, pred_len



class SEAttack(BaseAttack):
    def __init__(self, model, tokenizer, space_token, device, config):
        super(SEAttack, self).__init__(model, tokenizer, device, config, space_token)

        self.port_dict = {
            'de': 9000,
            'zh': 9001
        }

    def split_token(self, origin_target_sent):
        if self.target_language == 'zh':
            target_sent_seg = ' '.join(jieba.cut(origin_target_sent))
        else:
            target_sent_seg = ' '.join(origin_target_sent.split(' '))
        return target_sent_seg


class MyAttack(BaseAttack):
    def __init__(self, model, tokenizer, space_token, device, config):
        super(MyAttack, self).__init__(model, tokenizer, device, config, space_token)
        self.insert_character = string.punctuation
        self.insert_character += string.digits
        self.insert_character += string.ascii_letters

    def leave_eos_loss(self, scores, pred_len):
        loss = []
        for i, s in enumerate(scores):
            mask = torch.ones(s.shape[-1], device=s.device)
            mask[self.pad_token_id] = 1e-12
            s_safe = s * mask  # ✅ 完全避免 inplace
            eos_p = self.softmax(s_safe)[:pred_len[i], self.eos_token_id]
            loss.append(self.bce_loss(eos_p, torch.zeros_like(eos_p)))
        return loss


    def leave_eos_target_loss(self, scores, seqs, pred_len):
        loss = []
        for i, s in enumerate(scores):
            # clone to avoid inplace operation that would break autograd
            mask = torch.ones(s.shape[-1], device=s.device)
            mask[self.pad_token_id] = 1e-12
            s_safe = s * mask  # ✅ 完全避免 inplace

            softmax_v = self.softmax(s_safe)
            eos_p = softmax_v[:pred_len[i], self.eos_token_id]

            # 注意 s 可能比 target 短，所以截断
            target_p = torch.stack([
                softmax_v[j, token_id] for j, token_id in enumerate(seqs[i][1:])
            ])
            target_p = target_p[:pred_len[i]]

            pred = eos_p + target_p
            if pred.shape[0] > 0:
                pred[-1] = pred[-1] / 2  # safe inplace op on new tensor
            loss.append(self.bce_loss(pred, torch.zeros_like(pred)))
        return loss


    @torch.no_grad()
    def select_best(self, new_strings, batch_size=30):
        seqs = []
        batch_num = len(new_strings) // batch_size
        if batch_size * batch_num != len(new_strings):
            batch_num += 1
        for i in range(batch_num):
            st, ed = i * batch_size, min(i * batch_size + batch_size, len(new_strings))
            input_token = self.tokenizer(new_strings[st:ed], return_tensors="pt", padding=True).input_ids
            input_token = input_token.to(self.device)
            trans_res = translate(
                self.model, input_token,
                early_stopping=False, num_beams=self.num_beams,
                num_beam_groups=self.num_beam_groups, use_cache=True,
                max_length=self.max_len
            )
            seqs.extend(trans_res['sequences'].tolist())
        pred_len = np.array([self.compute_seq_len(torch.tensor(seq)) for seq in seqs])
        assert len(new_strings) == len(pred_len)
        return new_strings[pred_len.argmax()], max(pred_len)

    def prepare_attack(self, text):
        ori_len = self.get_trans_len(text)[0]      # int
        best_adv_text, best_len = deepcopy(text[0]), ori_len
        current_adv_text, current_len = deepcopy(text[0]), ori_len  # current_adv_text: List[str]
        return ori_len, (best_adv_text, best_len), (current_adv_text, current_len)

    def compute_loss(self, xxx):
        raise NotImplementedError

    def mutation(self, current_adv_text, grad, modify_pos):
        raise NotImplementedError

    def run_attack(self, text):
        assert len(text) == 1
        ori_len, (best_adv_text, best_len), (current_adv_text, current_len) = self.prepare_attack(text)
        adv_his = [(deepcopy(current_adv_text), deepcopy(current_len), 0.0)]
        modify_pos = []

        pbar = tqdm(range(self.max_per))
        t1 = time.time()
        for it in pbar:
            try:
                print(f"\n[DEBUG] Iteration {it}")

                # Step 1: forward 获取 scores
                scores, seqs, pred_len = self.compute_score([current_adv_text])

                # Step 2: 保留 score 的计算图以获取梯度
                for s in scores:
                    s.retain_grad()

                # Step 3: 用 score 算 loss
                loss_list = self.leave_eos_loss(scores, pred_len)  # 或换成 leave_eos_target_loss / untarget_loss
                print(f"[DEBUG] loss_list: {[l.item() for l in loss_list]}")
                loss = sum(loss_list)
                print(f"[DEBUG] total loss: {loss.item()}")

                # Step 4: 反向传播
                self.model.zero_grad()
                loss.backward()
                print("[DEBUG] Backward success")

                # Step 5: 获取 score 的梯度（以第一个样本为例）
                grad = scores[0].grad
                if grad is None:
                    print("[DEBUG] grad is None! Something went wrong during backward.")
                    return False, adv_his
                else:
                    print("[DEBUG] grad.shape:", grad.shape)

                # Step 6: 基于梯度进行扰动
                new_strings = self.mutation(current_adv_text, grad, modify_pos)
                print(f"[DEBUG] new_strings generated: {len(new_strings)}")

                if new_strings:
                    current_adv_text, current_len = self.select_best(new_strings)
                    log_str = "%d, %d, %.2f" % (it, len(new_strings), best_len / ori_len)
                    pbar.set_description(log_str)
                    print(f"[DEBUG] current_adv_text: {current_adv_text}, current_len: {current_len}")

                    if current_len > best_len:
                        best_adv_text = deepcopy(current_adv_text)
                        best_len = current_len
                    t2 = time.time()
                    adv_his.append((best_adv_text, int(best_len), t2 - t1))
                else:
                    print("[DEBUG] No new_strings returned. Attack stops.")
                    return False, adv_his


            except Exception as e:
                print("[EXCEPTION] Exception during run_attack:")
                import traceback
                traceback.print_exc()
            return False, adv_his

        print("[DEBUG] Attack completed.")
        return True, adv_his



class BaselineAttack(BaseAttack):
    def __init__(self, model, tokenizer, space_token, device, config):
        super(BaselineAttack, self).__init__(model, tokenizer, device, config, space_token)

        self.insert_character = string.punctuation
        self.insert_character += string.digits
        self.insert_character += string.ascii_letters

    def leave_eos_loss(self, scores, pred_len):
        loss = []
        for i, s in enumerate(scores):
            mask = torch.ones(s.shape[-1], device=s.device)
            mask[self.pad_token_id] = 1e-12
            s_safe = s * mask  # ✅ 完全避免 inplace
            eos_p = self.softmax(s_safe)[:pred_len[i], self.eos_token_id]
            loss.append(self.bce_loss(eos_p, torch.zeros_like(eos_p)))
        return loss


    def untarget_loss(self, scores, seqs, pred_len):
        loss = []
        for i, s in enumerate(scores):
            mask = torch.ones(s_safe.shape[-1], device=s_safe.device)
            mask[self.pad_token_id] = 1e-12  # 替换 pad token 的值
            # 应用到整个 batch，广播成 [seq_len, vocab_size]
            s_safe = s_safe * mask
            softmax_v = self.softmax(s_safe)
            target_p = torch.stack([softmax_v[iii, s] for iii, s in enumerate(seqs[i][1:])])
            target_p = target_p[:pred_len[i]]
            pred = target_p
            # pred[-1] = pred[-1] / 2
            loss.append(self.bce_loss(pred, torch.zeros_like(pred)))
        return loss



import copy
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import nltk
import string
from copy import deepcopy
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from .base_attack import MyAttack


class CharacterAttack(MyAttack):
    def __init__(self, model, tokenizer, space_token, device, config):
        super(CharacterAttack, self).__init__(model, tokenizer, space_token, device, config)

    def compute_loss(self, text):
        scores, seqs, pred_len = self.compute_score(text)
        loss_list = self.leave_eos_target_loss(scores, seqs, pred_len)
        return loss_list

    def mutation(self, current_adv_text, grad, modify_pos):
        current_tensor = self.tokenizer([current_adv_text], return_tensors="pt", padding=True).input_ids[0]
        new_strings = self.character_replace_mutation(current_adv_text, current_tensor, grad)
        return new_strings

    @staticmethod
    def transfer(c: str):
        if c in string.ascii_lowercase:
            return c.upper()
        elif c in string.ascii_uppercase:
            return c.lower()
        return c

    def character_replace_mutation(self, current_text, current_tensor, grad):
        important_tensor = (-grad.sum(1)).argsort()
        new_strings = [current_text]
        for t in important_tensor:
            if int(t) not in current_tensor:
                continue
            ori_decode_token = self.tokenizer.decode([int(t)])
            if self.space_token in ori_decode_token:
                ori_token = ori_decode_token.replace(self.space_token, '')
            else:
                ori_token = ori_decode_token
            if len(ori_token) == 1 or ori_token in self.specical_token:
                continue
            candidate = [ori_token[:i] + insert + ori_token[i:] for i in range(len(ori_token)) for insert in self.insert_character]
            candidate += [ori_token[:i - 1] + self.transfer(ori_token[i - 1]) + ori_token[i:] for i in range(1, len(ori_token))]

            new_strings += [current_text.replace(ori_token, c, 1) for c in candidate]
            if len(new_strings) != 0:
                return new_strings
        return new_strings


class WordAttack(MyAttack):
    def __init__(self, model, tokenizer, space_token, device, config):
        super(WordAttack, self).__init__(model, tokenizer, space_token, device, config)

    def mutation(self, current_adv_text, grad, modify_pos):
        new_strings = self.token_replace_mutation(current_adv_text, grad, modify_pos)
        return new_strings

    def compute_loss(self, text):
        scores, seqs, pred_len = self.compute_score(text)
        loss_list = self.leave_eos_target_loss(scores, seqs, pred_len)
        return loss_list

    def token_replace_mutation(self, current_adv_text, grad, modify_pos):
        new_strings = []
        current_tensor = self.tokenizer(current_adv_text, return_tensors="pt", padding=True).input_ids[0]
        base_tensor = current_tensor.clone()
        for pos in modify_pos:
            t = current_tensor[0][pos]
            grad_t = grad[t]
            score = (self.embedding - self.embedding[t]).mm(grad_t.reshape([-1, 1])).reshape([-1])
            index = score.argsort()
            for tgt_t in index:
                if tgt_t not in self.specical_token:
                    base_tensor[pos] = tgt_t
                    break
        current_text = self.tokenizer.decode(current_tensor)
        for pos, t in enumerate(current_tensor):
            if t not in self.specical_id:
                cnt, grad_t = 0, grad[t]
                score = (self.embedding - self.embedding[t]).mm(grad_t.reshape([-1, 1])).reshape([-1])
                index = score.argsort()
                for tgt_t in index:
                    if tgt_t not in self.specical_token:
                        new_base_tensor = base_tensor.clone()
                        new_base_tensor[pos] = tgt_t
                        candidate_s = self.tokenizer.decode(new_base_tensor)
                        new_strings.append(candidate_s)
                        cnt += 1
                        if cnt >= 50:
                            break
        return new_strings


class StructureAttack(MyAttack):
    def __init__(self, model, tokenizer, space_token, device, config):
        super().__init__(model, tokenizer, space_token, device, config)
        self.berttokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        self.bertmodel = BertForMaskedLM.from_pretrained('bert-large-uncased').to(device).eval()
        self.num_of_perturb = 50

    def compute_loss(self, text):
        scores, seqs, pred_len = self.compute_score(text)
        loss_list = self.leave_eos_target_loss(scores, seqs, pred_len)
        return loss_list

    def mutation(self, current_adv_text, grad, modify_pos):
        return self.structure_mutation(current_adv_text, grad)

    def get_token_type(self, input_tensor):
        tokens = self.tokenizer.convert_ids_to_tokens(input_tensor)
        tokens = [tk.replace(self.space_token, '') for tk in tokens]
        pos_inf = nltk.tag.pos_tag(tokens)
        return tokens, pos_inf

    def perturbBert(self, tokens, ori_tensor, masked_index):
        print("[DEBUG] >>> Entered perturbBert!")
        original_word = tokens[masked_index]
        tokens[masked_index] = '[MASK]'

        try:
            indexed_tokens = self.berttokenizer.convert_tokens_to_ids(tokens)
            tokens_tensor = torch.tensor([indexed_tokens]).to(self.model.device)
            logits = self.bertmodel(tokens_tensor).logits
        except Exception as e:
            print('[DEBUG] BERT prediction error:', e)
            return []

        topk_ids = torch.topk(logits[0, masked_index], self.num_of_perturb).indices.tolist()
        topk_tokens_raw = self.berttokenizer.convert_ids_to_tokens(topk_ids)
        print(f"[DEBUG] Top-k raw candidates: {topk_tokens_raw}")

        topk_tokens = [t for t in topk_tokens_raw if len(t) > 1 and t.isalpha()]
        print(f"[DEBUG] Filtered top-k tokens: {topk_tokens}")

        new_sentences = []
        for t in topk_tokens:
            print(f"[DEBUG] Trying replacement token: '{t}'")
            new_tokens = tokens[:]
            new_tokens[masked_index] = t
            try:
                new_sentence = self.berttokenizer.convert_tokens_to_string(new_tokens)
                print(f"[DEBUG] Generated sentence: {new_sentence}")
                new_sentences.append(new_sentence)
            except Exception as e:
                print(f"[DEBUG] Failed to convert '{t}' to string:", e)

        tokens[masked_index] = original_word
        print(f"[DEBUG] perturbBert: original word = {original_word}")
        print(f"[DEBUG] perturbBert: BERT top-{self.num_of_perturb} candidates = {topk_tokens}")
        print(f"[DEBUG] perturbBert: generated {len(new_sentences)} valid sentences")
        return new_sentences

    def structure_mutation(self, current_adv_text, grad):
        current_tensor = self.tokenizer(current_adv_text, return_tensors="pt", padding=True).input_ids[0].to(self.model.device)
        ori_tokens, pos_tags = self.get_token_type(current_tensor)

        if grad.shape[0] != len(current_tensor):
            print("[DEBUG] grad shape does not match token length. Adjusting gradient...")
            grad = grad.mean(dim=0, keepdim=True).repeat(len(current_tensor), 1)
            print(f"[DEBUG] Adjusted grad.shape: {grad.shape}")

        importance = []
        for i in range(len(current_tensor)):
            if ori_tokens[i] in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            score = torch.sum(torch.abs(grad[i])).item()
            importance.append((i, score))
        importance.sort(key=lambda x: x[1], reverse=True)
        print(f"[DEBUG] Token importance (top 5): {importance[:5]}")

        new_sentences = []
        for pos, score in importance:
            if pos == 0 or pos == len(current_tensor) - 1:
                continue
            perturbed = self.perturbBert(ori_tokens[:], current_tensor.clone(), pos)
            if perturbed:
                new_sentences.extend(perturbed)
            if len(new_sentences) > 2000:
                break

        print(f"[DEBUG] structure_mutation generated {len(new_sentences)} samples.")
        return new_sentences

    def compute_score(self, text):
        return super().compute_score(text)

    def leave_eos_target_loss(self, scores, seqs, pred_len):
        return super().leave_eos_target_loss(scores, seqs, pred_len)

from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model
#from eval import MultiWozEvaluator
from reader import MultiWozReader
import torch
import torch.nn as nn

import os
import random
import argparse
import time
import logging
import json
import numpy as np
import gdown

from config import global_config as cfg
# from config21 import global_config as cfg  # global, already initialized


import warnings
warnings.filterwarnings("ignore")


class Modal(object):
    def __init__(self, device):
        self.device = device
        # initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(cfg.gpt_path)
        # cfg.tokenizer = tokenizer

        # initialize multiwoz reader
        self.reader = MultiWozReader(self.tokenizer)
        output_path = './chatbotModel/'
        gdown.download("https://drive.google.com/file/d/1WaINRqDqaU2k7U75643qmPKDI3fJBmIG/view?usp=sharing", output_path, quiet=False)


        # create model: gpt2
        self.model = GPT2LMHeadModel.from_pretrained(output_path)
        if cfg.mode == 'train':
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)  # single gpu

        #
        #self.evaluator = MultiWozEvaluator(self.reader)
        self.tb_writer = None

    def get_optimizers(self):
        """
        Setup the optimizer and the learning rate scheduler.

        from transformers.Trainer

        parameters from cfg: lr (1e-3); warmup_steps
        """
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr)
        num_training_steps = self.reader.set_stats['train']['num_dials'] * \
                             cfg.epoch_num // (cfg.gradient_accumulation_steps*cfg.batch_size)
        num_warmup_steps = cfg.warmup_steps if cfg.warmup_steps >= 0 else int(num_training_steps*0.2)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        return optimizer, scheduler

    def log_first_inputs(self, inputs):
        tokenizer = self.tokenizer
        logging.info("**** Input Examples: ****")
        for context in inputs['contexts'][:4]:
            # ubar = tokenizer.convert_ids_to_tokens(context)
            # ubar = tokenizer.convert_tokens_to_string(context)
            # ubar = " ".join(ubar)
            ubar = tokenizer.decode(context)
            logging.info(ubar)

    def add_torch_input(self, inputs):
        # to tensor and to device
        contexts_tensor = torch.from_numpy(inputs['contexts_np']).long()
        contexts_tensor = contexts_tensor.to(self.device)
        inputs['contexts_tensor'] = contexts_tensor
        return inputs

    def add_torch_input_eval(self, inputs):
        # inputs: context
        inputs['context_tensor'] = torch.tensor(
            [inputs['context']]).to(self.device)
        return inputs

    def calculate_loss_and_accuracy(self, outputs, labels):
        # GPT2-chicahat/train.py
        lm_logits = outputs[0]

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        pad_id = cfg.pad_id
        loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # avg loss
        not_ignore = shift_labels.ne(pad_id)
        num_targets = not_ignore.long().sum().item()

        loss /= num_targets
        return loss

    def train_URURU(self):
        """
        URURU
        """
        all_batches = self.reader.get_batches('train')
        # compute num_training_steps in get_batches()
        optimizer, scheduler = self.get_optimizers()

        # log info
        set_stats = self.reader.set_stats['train']
        logging.info("***** Running training *****")
        logging.info("  Num Training steps(one turn in a batch of dialogs) per epoch = %d",
                     set_stats['num_training_steps_per_epoch'])
        logging.info("  Num Turns = %d", set_stats['num_turns'])
        logging.info("  Num Dialogs = %d", set_stats['num_dials'])
        logging.info("  Num Epochs = %d", cfg.epoch_num)
        logging.info("  Batch size  = %d", cfg.batch_size)
        logging.info("  Gradient Accumulation steps = %d",
                     cfg.gradient_accumulation_steps)
        logging.info("  Total optimization steps = %d",
                     set_stats['num_training_steps_per_epoch']*cfg.epoch_num // cfg.gradient_accumulation_steps)

        # tb writer
        if self.tb_writer is not None:
            self.tb_writer.add_text('cfg', json.dumps(cfg.__dict__, indent=2))
            # self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        log_inputs = 2
        global_step = 0
        sw = time.time()

        for epoch in range(cfg.epoch_num):
            epoch_step = 0
            tr_loss = 0.0
            logging_loss = 0.0
            btm = time.time()
            oom_time = 0
            self.model.zero_grad()

            data_iterator = self.reader.get_data_iterator(
                all_batches)

            for batch_idx, dial_batch in enumerate(data_iterator):
                pv_batch = None
                for turn_num, turn_batch in enumerate(dial_batch):
                    first_turn = (turn_num == 0)
                    inputs = self.reader.convert_batch_turn(turn_batch, pv_batch, first_turn)
                    pv_batch = inputs['labels']
                    try:  # avoid OOM
                        self.model.train()
                        if log_inputs > 0:  # log inputs for the very first two turns
                            self.log_first_inputs(inputs)
                            log_inputs -= 1

                        # to tensor
                        inputs = self.add_torch_input(inputs)
                        # loss
                        outputs = self.model(inputs['contexts_tensor'])
                        # outputs = self.model(inputs['contexts_tensor']) # debugging with GPT2Model
                        loss = self.calculate_loss_and_accuracy(
                            outputs, labels=inputs['contexts_tensor'])
                        loss.backward()
                        tr_loss += loss.item()
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 5.0)
                        epoch_step += 1

                        # step, wrt gradient_accumulation_steps, clip grad norm
                        if (epoch_step+1) % cfg.gradient_accumulation_steps == 0 or(
                                # end of an epoch
                                (epoch_step + \
                                 1) == set_stats['num_training_steps_per_epoch']
                        ):
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()
                            # global_step: actual step the optimizer took
                            global_step += 1

                            logs = {}  # for tb writer
                            # logging: loss, lr... after certain amount of steps
                            if cfg.report_interval > 0 and global_step % cfg.report_interval == 0:
                                loss_scalar = (tr_loss - logging_loss) / \
                                              cfg.report_interval
                                logging_loss = tr_loss
                                logs['loss'] = loss_scalar
                                logging.info(
                                    'Global step: {}, epoch step: {}, interval loss: {:.4f}'.format(
                                        global_step, epoch_step, loss_scalar
                                    ))
                                # validate
                                # add to tensorboard...
                                if cfg.evaluate_during_training and loss_scalar < 10:
                                    results = self.validate()
                                    for k, v in results.items():
                                        eval_key = "eval_{}".format(k)
                                        logs[eval_key] = v

                                if self.tb_writer:
                                    for k, v in logs.items():
                                        self.tb_writer.add_scalar(
                                            k, v, global_step)
                                # save model... 

                    except RuntimeError as exception:
                        if "out of memory" in str(exception):
                            max_length = max(inputs['lengths'])
                            oom_time += 1
                            logging.info("WARNING: ran out of memory,times: {}, batch size: {}, max_len: {}".format(
                                oom_time, cfg.batch_size, max_length))
                            if hasattr(torch.cuda, 'empty_cache'):
                                torch.cuda.empty_cache()
                        else:
                            logging.info(str(exception))
                            raise exception
            logging.info('Train epoch time: {:.2f} min, epoch loss: {:.4f}'.format(
                (time.time()-btm)/60, tr_loss))
            # save model after every epoch
            # if epoch > 30 and tr_loss/epoch_step < 0.6:
            self.save_model(epoch, tr_loss/epoch_step)




    def train(self):
        """
        UBARU
        """
        all_batches = self.reader.get_batches('train')
        # compute num_training_steps in get_batches()
        optimizer, scheduler = self.get_optimizers()

        # log info
        set_stats = self.reader.set_stats['train']
        logging.info("***** Running training *****")
        logging.info("  Num Training steps(one turn in a batch of dialogs) per epoch = %d",
                     set_stats['num_training_steps_per_epoch'])
        logging.info("  Num Turns = %d", set_stats['num_turns'])
        logging.info("  Num Dialogs = %d", set_stats['num_dials'])
        logging.info("  Num Epochs = %d", cfg.epoch_num)
        logging.info("  Batch size  = %d", cfg.batch_size)
        logging.info("  Gradient Accumulation steps = %d",
                     cfg.gradient_accumulation_steps)
        logging.info("  Total optimization steps = %d",
                     set_stats['num_dials']*cfg.epoch_num // (cfg.gradient_accumulation_steps*cfg.batch_size))

        # tb writer
        if self.tb_writer is not None:
            self.tb_writer.add_text('cfg', json.dumps(cfg.__dict__, indent=2))
            # self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        log_inputs = 2
        global_step = 0
        sw = time.time()

        for epoch in range(cfg.epoch_num):
            epoch_step = 0
            tr_loss = 0.0
            logging_loss = 0.0
            btm = time.time()
            oom_time = 0
            self.model.zero_grad()

            data_iterator = self.reader.get_nontranspose_data_iterator(
                all_batches)

            for batch_idx, dial_batch in enumerate(data_iterator):
                inputs = self.reader.convert_batch_session(dial_batch)
                try:  # avoid OOM
                    self.model.train()
                    if log_inputs > 0:  # log inputs for the very first two turns
                        self.log_first_inputs(inputs)
                        log_inputs -= 1

                    # to tensor
                    inputs = self.add_torch_input(inputs)
                    # loss
                    outputs = self.model(inputs['contexts_tensor'])
                    # outputs = self.model(inputs['contexts_tensor']) # debugging with GPT2Model
                    loss = self.calculate_loss_and_accuracy(
                        outputs, labels=inputs['contexts_tensor'])
                    loss.backward()
                    tr_loss += loss.item()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 5.0)
                    epoch_step += 1

                    # step, wrt gradient_accumulation_steps, clip grad norm
                    if (epoch_step+1) % cfg.gradient_accumulation_steps == 0 or(
                            # end of an epoch
                            (epoch_step + \
                             1) == set_stats['num_training_steps_per_epoch']
                    ):
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        # global_step: actual step the optimizer took
                        global_step += 1

                        logs = {}  # for tb writer
                        # logging: loss, lr... after certain amount of steps
                        if cfg.report_interval > 0 and global_step % cfg.report_interval == 0:
                            loss_scalar = (tr_loss - logging_loss) / \
                                          cfg.report_interval
                            logging_loss = tr_loss
                            logs['loss'] = loss_scalar
                            logging.info(
                                'Global step: {}, epoch step: {}, interval loss: {:.4f}'.format(
                                    global_step, epoch_step, loss_scalar
                                ))
                            # validate
                            # add to tensorboard...
                            if cfg.evaluate_during_training and loss_scalar < 10:
                                results = self.validate()
                                for k, v in results.items():
                                    eval_key = "eval_{}".format(k)
                                    logs[eval_key] = v

                            if self.tb_writer:
                                for k, v in logs.items():
                                    self.tb_writer.add_scalar(
                                        k, v, global_step)
                            # save model... 

                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        max_length = max(inputs['lengths'])
                        oom_time += 1
                        logging.info("WARNING: ran out of memory,times: {}, batch size: {}, max_len: {}".format(
                            oom_time, cfg.batch_size, max_length))
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        logging.info(str(exception))
                        raise exception
            logging.info('Train epoch time: {:.2f} min, epoch loss: {:.4f}'.format(
                (time.time()-btm)/60, tr_loss))
            # save model after every epoch
            # if epoch > 10 or tr_loss/epoch_step < 1:
            self.save_model(epoch, tr_loss/epoch_step)

    def save_model(self, epoch, loss):
        save_path = os.path.join(
            cfg.exp_path, 'epoch{}_trloss{:.2f}_gpt2'.format(epoch+1, loss))
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        logging.info('Saving model checkpoint to %s', save_path)
        # save gpt2
        self.model.save_pretrained(save_path)
        # save tokenizer
        self.tokenizer.save_pretrained(save_path)
        # save cfg

    # def validate_URURU(self, data='dev', do_test=False):
    #     # predict one dialog/ one turn at a time
    #     self.model.eval()

    #     # all_batches = self.reader.get_batches('dev')
    #     # data_iterator = self.reader.get_data_iterator(all_batches)
    #     eval_data = self.reader.get_eval_data(data)

    #     set_stats = self.reader.set_stats[data]
    #     logging.info("***** Running Evaluation *****")
    #     logging.info("  Num Turns = %d", set_stats['num_turns'])
    #     # logging.info("  Num Dialogs = %d", set_stats['num_dials'])

    #     # valid_losses = []
    #     btm = time.time()
    #     result_collection = {}
    #     with torch.no_grad():
    #         eval_pbar = eval_data
    #         for dial_idx, dialog in enumerate(eval_pbar):

    #             pv_turn = {}
    #             for turn_idx, turn in enumerate(dialog):
    #                 first_turn = (turn_idx == 0)
    #                 inputs = self.reader.convert_turn_eval_URURU(
    #                     turn, pv_turn, first_turn)
    #                 inputs = self.add_torch_input_eval(inputs)

    #                 # fail to generate new tokens, if max_length not set
    #                 context_length = len(inputs['context'])
    #                 if cfg.use_true_curr_bspn: # generate act, response
    #                     max_len=60
    #                     if not cfg.use_true_curr_aspn:
    #                         max_len = 80

    #                     outputs = self.model.generate(input_ids=inputs['context_tensor'],
    #                                                   max_length=context_length+max_len, temperature=0.7, # top_p=0.9, num_beams=4,
    #                                                   pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.encode(['<eos_r>'])[0])
    #                     #   no_repeat_ngram_size=4
    #                     # turn['generated'] = self.tokenizer.decode(outputs[0])

    #                     # resp_gen, need to trim previous context
    #                     generated = outputs[0].cpu().numpy().tolist()
    #                     generated = generated[context_length-1:]

    #                     try:
    #                         decoded = self.decode_generated_act_resp(generated)
    #                     except ValueError as exception:
    #                         logging.info(str(exception))
    #                         logging.info(self.tokenizer.decode(generated))
    #                         decoded = {'resp': [], 'bspn': [], 'aspn': []}

    #                 else: # predict bspn, access db, then generate act and resp
    #                     outputs = self.model.generate(input_ids=inputs['context_tensor'],
    #                                                   max_length=context_length+60, temperature=0.7, # top_p=0.9, num_beams=4,
    #                                                   pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.encode(['<eos_b>'])[0])
    #                     generated_bs = outputs[0].cpu().numpy().tolist()
    #                     # generated_bs = generated_bs[context_length-1:]
    #                     bspn_gen = self.decode_generated_bspn(generated_bs[context_length-1:])
    #                     # check DB result
    #                     if cfg.use_true_db_pointer:
    #                         # db_result = self.reader.bspan_to_DBpointer(self.tokenizer.decode(turn['bspn']), turn['turn_domain'])
    #                         db = turn['db']
    #                     else:
    #                         db_result = self.reader.bspan_to_DBpointer(self.tokenizer.decode(bspn_gen), turn['turn_domain'])
    #                         db = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('<sos_db> '+ db_result + ' <eos_db>')) + self.tokenizer.encode(['<sos_a>'])
    #                     inputs['context_tensor_db'] = torch.tensor([inputs['context'][:-1] + bspn_gen + db]).to(self.device)
    #                     context_length = len(inputs['context_tensor_db'][0])
    #                     outputs_db = self.model.generate(input_ids=inputs['context_tensor_db'],
    #                                                      max_length=context_length+80, temperature=0.7, # top_p=0.9, num_beams=4,
    #                                                      pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.encode(['<eos_r>'])[0])
    #                     generated_ar = outputs_db[0].cpu().numpy().tolist()
    #                     generated_ar = generated_ar[context_length-1:]
    #                     try:
    #                         decoded = self.decode_generated_act_resp(generated_ar)
    #                         decoded['bspn'] = bspn_gen
    #                     except ValueError as exception:
    #                         logging.info(str(exception))
    #                         logging.info(self.tokenizer.decode(generated_ar))
    #                         decoded = {'resp': [], 'bspn': [], 'aspn': []}

    #                 turn['resp_gen'] = decoded['resp']
    #                 turn['bspn_gen'] = turn['bspn'] if cfg.use_true_curr_bspn else decoded['bspn']
    #                 turn['aspn_gen'] = turn['aspn'] if cfg.use_true_curr_aspn else decoded['aspn']
    #                 turn['dspn_gen'] = turn['dspn']

    #                 # check DB results
    #                 # db_result = self.reader.bspan_to_DBpointer(self.tokenizer.decode(turn['bspn']), turn['turn_domain'])
    #                 # if db_result[0] == 1: # no match
    #                 #     print('gt:', self.tokenizer.decode(turn['aspn']), '     |gen:', self.tokenizer.decode(decoded['aspn']))
    #                 #     print('gen_resp: ', self.tokenizer.decode(decoded['resp']))
    #                 #     print('gt_resp: ', self.tokenizer.decode(turn['resp']), '\n')

    #                 pv_turn['labels'] = inputs['labels'] # all true previous context
    #                 pv_turn['resp'] = turn['resp'] if cfg.use_true_prev_resp else decoded['resp']
    #                 # pv_turn['bspn'] = turn['bspn'] if cfg.use_true_prev_bspn else decoded['bspn']
    #                 # pv_turn['db'] = db
    #                 # pv_turn['aspn'] = turn['aspn'] if cfg.use_true_prev_aspn else decoded['aspn']
    #                 # pv_turn = inputs['labels']

    #             result_collection.update(
    #                 self.reader.inverse_transpose_turn(dialog))

    #     logging.info("inference time: {:.2f} min".format((time.time()-btm)/60))
    #     # score
    #     btm = time.time()
    #     results, _ = self.reader.wrap_result_lm(result_collection)
    #     bleu, success, match = self.evaluator.validation_metric(results)
    #     logging.info("Scoring time: {:.2f} min".format((time.time()-btm)/60))
    #     score = 0.5 * (success + match) + bleu
    #     valid_loss = 130 - score
    #     logging.info('validation [CTR] match: %2.2f  success: %2.2f  bleu: %2.2f    score: %.2f' % (
    #         match, success, bleu, score))
    #     eval_results = {}
    #     eval_results['bleu'] = bleu
    #     eval_results['success'] = success
    #     eval_results['match'] = match

    #     return eval_results

    def validate(self, data='dev', pv_turn =
    {'labels': [31373, 1312, 765, 284, 1492, 257, 7541, 2119, 50314, 50308, 50317, 220, 50319, 220, 50318, 50315, 50315, 50264, 50274, 50264, 50275, 50309, 50313, 8788, 837, 1049, 764, 1312, 716, 9675, 1312, 714, 1037, 345, 351, 326, 764, 460, 345, 1560, 502, 644, 345, 561, 588, 284, 2652, 379, 5633, 50307, 72, 765, 284, 1492, 257, 2119, 220], 'resp': [50313, 1312, 716, 7926, 475, 1312, 716, 407, 4917, 257, 50281, 326, 7466, 534, 2476, 764, 561, 345, 588, 502, 284, 804, 287, 1194, 1989, 5633, 50307], 'bspn': [50314, 50308], 'db': [50317, 220, 50319, 220, 50318, 50315], 'aspn': [50315, 50264, 50274, 50264, 50275, 50309]}
                 , my_string = "hi"):
        # predict one dialog/ one turn at a time
        #self.model.eval()

        set_stats = self.reader.set_stats[data]

        #first_turn = (turn_idx == 0)
        my_string = input("Enter a string: ")

        input_ids = self.tokenizer.encode(my_string)
        turn = { 'user': input_ids}
        inputs = self.reader.convert_turn_eval(
            turn, pv_turn, False)

        #print(turn)
        inputs = self.add_torch_input_eval(inputs)
        first_turn = False
        # fail to generate new tokens, if max_length not set
        context_length = len(inputs['context'])
        outputs = self.model.generate(input_ids=inputs['context_tensor'],
                                          max_length=context_length+60, temperature=0.7, # top_p=0.9, num_beams=4,
                                          pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.encode(['<eos_b>'])[0])
        generated_bs = outputs[0].cpu().numpy().tolist()
        # generated_bs = generated_bs[context_length-1:]
        bspn_gen = self.decode_generated_bspn(generated_bs[context_length-1:])
        # check DB result
        if cfg.use_true_db_pointer:
            # db_result = self.reader.bspan_to_DBpointer(self.tokenizer.decode(turn['bspn']), turn['turn_domain'])
            db = turn['db']
        else:
            db_result = self.reader.bspan_to_DBpointer(self.tokenizer.decode(bspn_gen), ['[hotel]'])
            print(db_result)
            print(self.tokenizer.decode(bspn_gen))
            db = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('<sos_db> '+ db_result + ' <eos_db>')) + self.tokenizer.encode(['<sos_a>'])
        inputs['context_tensor_db'] = torch.tensor([inputs['context'][:-1] + bspn_gen + db]).to(self.device)
        context_length = len(inputs['context_tensor_db'][0])
        outputs_db = self.model.generate(input_ids=inputs['context_tensor_db'],
                                         max_length=context_length+80, temperature=0.7, # top_p=0.9, num_beams=4,
                                         pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.encode(['<eos_r>'])[0])
        generated_ar = outputs_db[0].cpu().numpy().tolist()
        generated_ar = generated_ar[context_length-1:]
        try:
            decoded = self.decode_generated_act_resp(generated_ar)
            decoded['bspn'] = bspn_gen

        except ValueError as exception:
            logging.info(str(exception))
            logging.info(self.tokenizer.decode(generated_ar))
            decoded = {'resp': [], 'bspn': [], 'aspn': []}
        res = self.tokenizer.decode(decoded['resp'])
        bspn = self.tokenizer.decode(decoded['bspn'])
        aspn = self.tokenizer.decode(decoded['aspn'])


        turn['resp_gen'] = decoded['resp']
        turn['bspn_gen'] = turn['bspn'] if cfg.use_true_curr_bspn else decoded['bspn']
        turn['aspn_gen'] = turn['aspn'] if cfg.use_true_curr_aspn else decoded['aspn']
        turn['dspn_gen'] = [50316, 220, 50258, 220, 50311]

        # check DB results

        pv_turn['labels'] = inputs['labels'] # all true previous context
        pv_turn['resp'] = turn['resp'] if cfg.use_true_prev_resp else decoded['resp']
        pv_turn['bspn'] = turn['bspn'] if cfg.use_true_prev_bspn else decoded['bspn']
        #print(self.tokenizer.decode(pv_turn['bspn']))
        pv_turn['db'] = turn['db'] if cfg.use_true_curr_bspn else db
        pv_turn['aspn'] = turn['aspn'] if cfg.use_true_prev_aspn else decoded['aspn']
        result = {
            'pv_turn': pv_turn,
            'response': res,
            'bspn': bspn,
            'aspn': aspn
        }
             
        return result

    def decode_generated_act_resp(self, generated):
        """
        decode generated
        return decoded['resp'] ('bspn', 'aspn')
        """
        decoded = {}
        eos_a_id = self.tokenizer.encode(['<eos_a>'])[0]
        eos_r_id = self.tokenizer.encode(['<eos_r>'])[0]
        eos_b_id = self.tokenizer.encode(['<eos_b>'])[0]

        # eos_r may not exists if gpt2 generated repetitive words.
        if eos_r_id in generated:
            eos_r_idx = generated.index(eos_r_id)
        else:
            eos_r_idx = len(generated)-1
            logging.info('eos_r not in generated: ' + self.tokenizer.decode(generated))
        # eos_r_idx = generated.index(eos_r_id) if eos_r_id in generated else len(generated)-1

        if cfg.use_true_curr_aspn:  # only predict resp
            decoded['resp'] = generated[: eos_r_idx+1]
        else:  # predicted aspn, resp
            eos_a_idx = generated.index(eos_a_id)
            decoded['aspn'] = generated[: eos_a_idx+1]
            decoded['resp'] = generated[eos_a_idx+1: eos_r_idx+1]
        # if cfg.use_true_curr_bspn:

        # else:  # predict bspn aspn resp
        #     eos_b_idx = generated.index(eos_b_id)
        #     eos_a_idx = generated.index(eos_a_id)
        #     decoded['bspn'] = generated[: eos_b_idx+1]
        #     decoded['aspn'] = generated[eos_b_idx+1: eos_a_idx+1]
        #     decoded['resp'] = generated[eos_a_idx+1: eos_r_idx+1]
        return decoded

    def decode_generated_bspn(self, generated):
        eos_b_id = self.tokenizer.encode(['<eos_b>'])[0]
        if eos_b_id in generated:
            eos_b_idx = generated.index(eos_b_id)
        else:
            eos_b_idx = len(generated)-1
        return generated[: eos_b_idx+1]

def parse_arg_cfg(args):
    # add args to cfg
    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            dtype = type(getattr(cfg, k))
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            elif dtype is list:
                v = v.split(',')
                if k == 'cuda_device':
                    v = [int(no) for no in v]
            else:
                v = dtype(v)
            setattr(cfg, k, v)
    return


def start():
    global device
    global m
    if not os.path.exists('./experiments'):
        os.mkdir('./experiments')

    if not os.path.exists('./experiments_21'):
        os.mkdir('./experiments_21')

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()

    cfg.mode = args.mode
    args.mode = "test"
    parse_arg_cfg(args)
    # cfg.model_path = cfg.eval_load_path
    cfg.gpt_path = cfg.eval_load_path

    cfg._init_logging_handler(args.mode)
    
    # fix random seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device('cpu')
    logging.info('Device: {}'.format(format(device)))
    # initialize model
    m = Modal(device)

    
    logging.info("Generate setting: \n\t use true_prev_bspn={} \n\t use true_prev_aspn={} \n\t use true_db_pointer={} \n\t use true_prev_resp={} \n\t use true_curr_bspn={} \n\t use true_curr_aspn={} \n\t use_all_previous_context={}".format(
            cfg.use_true_prev_bspn, cfg.use_true_prev_aspn, cfg.use_true_db_pointer, cfg.use_true_prev_resp,
            cfg.use_true_curr_bspn, cfg.use_true_curr_aspn, cfg.use_all_previous_context
        ))


def validate(mystring):
    return m.validate(mystring = mystring)



if __name__ == "__main__":
    start()

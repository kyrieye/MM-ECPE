import os
import time

class VocabConfig:
    init_word2idx = { '<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3 }
    embedding_size = 300
    pretrained = 'GloVe'


class MSVDLoaderConfig:
    train_caption_fpath = "dataset/MSVD/metadata/train.csv"
    val_caption_fpath = "dataset/MSVD/metadata/val.csv"
    test_caption_fpath = "dataset/MSVD/metadata/test.csv"

    em_train_caption_fpath = "dataset/EmVidCap-S/EmVideo_trainval_captions.csv"
    em_test_caption_fpath = "dataset/EmVidCap-S/EmVideo_test_captions.csv"
    # em_train_caption_fpath = "dataset/EmVidCap-S/EmVideo_trainval_captions.csv"
    # em_test_caption_fpath = "dataset/EmVidCap-S/EmVideo_test_captions.csv"

    min_count = 1
    max_caption_len = 15

    split_video_feat_fpath_tpl = "dataset/{}/features/{}_{}.hdf5"
    frame_sample_len = 30

    split_video_feat_fpath_tpl_em = "dataset/EmVidCap-S/features/{}_{}.hdf5"
    # split_video_feat_fpath_tpl_em = "dataset/EmVidCap-S/features/{}_{}.hdf5"

    split_negative_vids_fpath = "dataset/MSVD/metadata/neg_vids_{}.json" 
    split_negative_emvids_fpath = "dataset/MSVD/metadata/S_v2v/`neg_Emvids_{}_S`.json"
    # split_negative_emvids_fpath = "dataset/MSVD/metadata/Combine_v2v/neg_Emvids_{}_Combine.json"  

    num_workers = 1


class VisualEncoderConfig:
    app_feat, app_feat_size = 'clip', 512
    mot_feat, mot_feat_size = 'clip', 512
    feat_size = app_feat_size

class Emo_transConfig:
    num_layers = 4
    num_heads = 4
    dim_k = 32
    dim_v = 32
    dim_inner = 512
    dropout = 0.1


class PhraseEncoderConfig:
    SA_num_layers = 1; assert SA_num_layers == 1
    SA_num_heads = 1; assert SA_num_heads == 1
    SA_dim_k = 32
    SA_dim_v = 32
    SA_dim_inner = 512
    SA_dropout = 0.1


class DecoderConfig:
    sem_align_hidden_size = 512
    sem_attn_hidden_size = 512
    rnn_num_layers = 1
    rnn_hidden_size = 512
    max_teacher_forcing_ratio = 1.0
    min_teacher_forcing_ratio = 1.0


class Config:
    seed = 0

    corpus = 'MSVD' 
    
    pretrained_fpath = "/home/yec/Video_Cap/EPAN-main_subj/checkpoints/EmVidCap-S/30.ckpt"
    # pretrained_fpath = "/home/yec/Video_Cap/EPAN-main_subj/checkpoints/EmVidCap-S/11.ckpt"
    vocab = VocabConfig
    loader = MSVDLoaderConfig
    vis_encoder = VisualEncoderConfig
    phr_encoder = PhraseEncoderConfig
    decoder = DecoderConfig
    emo_transformer = Emo_transConfig

    """ Optimization """
    epochs = 30
    gradient_clip = 5.0 # None if not used
    PS_threshold = 0.2  
    model_name = "kpyu/video-blip-flan-t5-xl-ego4d"
    cause_prompt = "Please generate factual and emotional description respectively based on given video contents, cause features, and emotion features."
    caption_prompt = "Please generate an emotional captions for this video based on the following factual and emotional semantics: "
    device_ids = [0,1,2,3]
    emo_num = 179    
    batch_size = 2
    lr = 0.0007
    CA_beta = 1
    CA_lambda = 0.2
    lambda_1 = 0.2
    lambda_2 = 1
    em_wei = 0.1
    mode = "debug"
    file_path = 'EmVidCap-S'
    # file_path = 'EmVidCap-S'

    """ Evaluation """
    metrics_full = ['Bleu_1', 'Bleu_2', 'Bleu_3',  'Bleu_4', 'CIDEr', 'ROUGE_L', 'Acc_sw', 'Acc_c', 'BFS', 'CFS' ]

    """ Log """
    log_dpath = os.path.join("logs", file_path )
    ckpt_fpath_tpl = os.path.join("checkpoints", file_path, "{}.ckpt")
    ckpt_fpath_tpl_em = os.path.join("checkpoints", file_path, "{}.ckpt")
    result_dir = os.path.join("results", file_path)
    score_fpath = os.path.join( result_dir, "results.txt")
    score_fpath_stage1 = os.path.join( result_dir, "results_stage1.txt")

    """ TensorboardX """
    tx_train_loss = "loss/train"
    tx_train_cross_entropy_loss = "loss/train/cross_entropy"
    tx_train_contrastive_attention_loss = "loss/train/contrastive_attention"
    tx_train_cls_loss =  "loss/train/cls_loss"
    tx_val_loss = "loss/val"
    tx_val_cross_entropy_loss = "loss/val/cross_entropy"
    tx_val_contrastive_attention_loss = "loss/val/contrastive_attention"
    tx_val_cls_loss = "loss/val/cls_loss"
    tx_lr = "params/lr"
    tx_teacher_forcing_ratio = "params/teacher_forcing_ratio"



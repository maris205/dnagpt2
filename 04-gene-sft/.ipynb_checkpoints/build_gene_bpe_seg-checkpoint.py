import sentencepiece as spm

spm.SentencePieceTrainer.train(input='../01-data_env/data/dna_1g.txt,../01-data_env/data/protein_1g.txt',
                               model_prefix='gene_bpe_seg', 
                               vocab_size=60000,
                               model_type='bpe', #默认是unigram
                               num_threads=10,
                              )

from sentencepiece import SentencePieceProcessor
model_path = "gene_bpe_seg.model"
sp_model = SentencePieceProcessor(model_file=model_path)
mm = sp_model.EncodeAsPieces("TCGACGGCACGCGACAGCAGCGAGCCCCGCGCACCCGAGCGCGAKCGFVGPMVHLKVHLEADVASSCRSAVIYLTSEEPFEGVLGLRLKEGIAITGCWPRWPDEMDERSAVWRVEPYTRHFGRVLYSFGV")
print(mm)
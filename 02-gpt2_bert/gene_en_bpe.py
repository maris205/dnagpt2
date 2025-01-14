from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False) #use_regex=False,空格当成一般字符串
trainer = trainers.BpeTrainer(vocab_size=90000, special_tokens=["<|endoftext|>"]) #9w words

tokenizer.train(["dna_1g.txt","protein_1g.txt","english_500m.txt"]
                , trainer=trainer) #all file list, take 10-20 min


tokenizer.save("gene_eng_dict.json")
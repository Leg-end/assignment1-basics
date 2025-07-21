from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path, VOCAB_PATH, MERGES_PATH, FIXTURES_PATH

def test_overlapping_special_tokens():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
        special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"],
    )
    print(VOCAB_PATH)
    print(MERGES_PATH)
    # test_string = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"

    # ids = tokenizer.encode(test_string)
    # tokenized_string = [tokenizer.decode([x]) for x in ids]
    # # Ensure the double <|endoftext|><|endoftext|> is preserved as a single token
    # assert tokenized_string.count("<|endoftext|>") == 1
    # assert tokenized_string.count("<|endoftext|><|endoftext|>") == 1
    # # Test roundtrip
    # assert tokenizer.decode(ids) == test_string


if __name__ == "__main__":
    test_overlapping_special_tokens()
import torch

def get_token_embeddings(sentence, tokenizer, model, device):
    inputs = tokenizer(sentence, return_tensors="pt", return_attention_mask=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    token_embeddings = outputs.last_hidden_state.squeeze(0)[1:-1]  # remove <s> and </s>
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())[1:-1]

    return tokens, token_embeddings

def collapse_tokens(tokens, embeddings):
    words = []
    word_embeds = []
    current_word = ""
    current_vecs = []

    for tok, vec in zip(tokens, embeddings):
        clean_tok = tok.replace("▁", "")
        if "▁" in tok and current_word:
            words.append(current_word)
            word_embeds.append(torch.stack(current_vecs).mean(dim=0))
            current_word = clean_tok
            current_vecs = [vec]
        else:
            current_word += clean_tok
            current_vecs.append(vec)

    if current_word:
        words.append(current_word)
        word_embeds.append(torch.stack(current_vecs).mean(dim=0))

    return words, torch.stack(word_embeds)
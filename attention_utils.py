def flatten(l):
    return [item for sublist in l for item in sublist]


def remove_tech_tokens(mystr, tokens_to_remove=['<eos>', '<sos>', '<unk>', '<pad>']):
    return [x for x in mystr if x not in tokens_to_remove]


def get_text(x, vocab):
    text = [vocab.itos[token] for token in x]
    try:
        end_idx = text.index('<eos>')
        text = text[:end_idx]
    except ValueError:
        pass
    text = remove_tech_tokens(text)
    if len(text) < 1:
        text = []
    return text


def generate_translation(src, src_len, trg, model, TRG_vocab, SRC_vocab):
    model.eval()

    output = model(src, src_len, trg, 0)  # turn off teacher forcing
    output = output.argmax(dim=-1).cpu().numpy()

    original_ru = get_text(list(src[:, 0].cpu().numpy()), SRC_vocab)
    original_en = get_text(list(trg[:, 0].cpu().numpy()), TRG_vocab)
    generated_en = get_text(list(output[1:, 0]), TRG_vocab)

    print('Russian: {}'.format(' '.join(original_ru)))
    print('English: {}'.format(' '.join(original_en)))
    print('Generated: {}'.format(' '.join(generated_en)))
    print()
import streamlit as st 
from transformers import BartTokenizer, BartForConditionalGeneration

def summarize(text, max_length=500, min_length=40):
    model_name = "facebook/bart-large-cnn"
    cd = "E:/text summarixer huggingface"
    tokenizer = BartTokenizer.from_pretrained(model_name, cache_dir=cd)
    model = BartForConditionalGeneration.from_pretrained(model_name, cache_dir=cd)
    input_ids = tokenizer([text], max_length=max_length, return_tensors='pt', truncation=True)
    #generating ids for summary
    summary_ids = model.generate(input_ids['input_ids'], max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def main():
    st.title('Text-summarizer')
    inp = st.text_area('Enter the text')
    if st.button('Summarize'):
        if inp:
            summary = summarize(inp)
            st.subheader('Summary')
            st.write(summary)

if __name__ == "__main__":
    main()
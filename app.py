import streamlit as st
import json
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

st.set_page_config(page_title="GTU On Lisans ve Lisans Yonergesi Soru Cevap Sistemi")
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stAppDeployButton {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

@st.cache_resource
def load_model():
    try:
        model = AutoModelForQuestionAnswering.from_pretrained("./gtu_qa_model2")
        tokenizer = AutoTokenizer.from_pretrained("./gtu_qa_model2")
        return pipeline("question-answering", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None

@st.cache_data
def load_sections():
    try:
        with open('data/sections.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading sections: {str(e)}")
        return []

def main():
    st.title("GTU On Lisans ve Lisans Yonergesi Soru Cevap Sistemi")
    model = load_model()
    if model is None:
        st.error("Failed to load model")
        return
    
    sections = load_sections()
    if not sections:
        st.error("Failed to load sections")
        return
    
    section_names = [section['section'] for section in sections]
    selected_section = st.selectbox("Select Section", section_names)
    selected_context = next(section['context'] for section in sections if section['section'] == selected_section)
    st.text_area("Context", value=selected_context, height=200, disabled=True)
    question = st.text_input("Enter your question:")
    
    if st.button("Get Answer"):
        if question:
            try:
                result = model(question=question, context=selected_context)
                
                # if result['score'] < 0.01:
                #     st.warning("Üzgünüm, bu soruya net bir cevap veremiyorum. Verilen bağlamda sorunuzu tam olarak yanıtlayabilecek yeterli bilgi bulamadım. Lütfen sorunuzu yeniden düzenleyip tekrar deneyiniz.")
                # else:
                st.success(f"Answer: {result['answer']}")
                st.info(f"Confidence: {result['score']:.2%}")
            except Exception as e:
                st.error(f"Error during inference: {str(e)}")
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
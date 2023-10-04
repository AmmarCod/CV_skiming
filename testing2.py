import os
import gradio as gr
from pdfminer.high_level import extract_text
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
from langdetect import detect
from googletrans import Translator
import torch
import docx
import webbrowser

# Define the URL you want to open in the web browser


# nltk.download('punkt')
# # Download NLTK stopwords if not already downloaded
# nltk.download('stopwords')

# Load a pre-trained model from the Hugging Face Hub
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Create a summarizer pipeline using the sshleifer/distilbart-cnn-12-6 model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
# summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


def extract_text_from_docx(file_path):
    try:
        doc = docx.Document(file_path)
        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        raise Exception(f"Error extracting text from Word document: {str(e)}")
        
        
# Define a function for text preprocessing
def preprocess_text(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]

    # Join the tokens back into a single string
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

# Define a function that takes a list of files and a description as inputs and returns a list of outputs
def read_pdf(files, description):
    outputs = []
    cv_info = []  # Store CV information along with similarity scores

    total_pdfs = len(files)  # Count the total number of PDFs uploaded

    for file in files:
        try:
            file_extension = os.path.splitext(file.name)[-1].lower()

            if file_extension == '.pdf':
                # Read the text from the PDF file
                text = extract_text(file.name)
            elif file_extension == '.docx':
                # Read the text from the Word document
                text = extract_text_from_docx(file.name)
            else:
                raise Exception(f"Unsupported file format: {file_extension}")

            if not text:
                raise Exception("Document extraction failed or returned empty text.")
            # Detect the language of the text (with error handling)
            try:
                detected_language = detect(text)
            except Exception as language_detection_error:
                detected_language = "en"  # Default to English in case of error

            # Translate text to English if it's not in English (with error handling)
            try:
                if detected_language != "en":
                    translator = Translator()
                    text = translator.translate(text, dest="en").text
            except Exception as translation_error:
                pass  # If translation fails, proceed with the original text

            # Limit the input text length
            max_input_length = 1000  # Define a maximum length
            if len(text) > max_input_length:
                text = text[:max_input_length]

            # Preprocess the text
            preprocessed_text = preprocess_text(text)

            # Generate the summary in English
            summary = summarizer(preprocessed_text, min_length=50, max_length=1000)[0]["summary_text"]
            
            # Translate the summary to English if needed (with error handling)
            try:
                if detected_language != "en":
                    summary_translator = Translator()
                    summary = summary_translator.translate(summary, dest="en").text
            except Exception as summary_translation_error:
                pass  # If translation fails, proceed with the original summary

            embeddings1 = model.encode(preprocessed_text, convert_to_tensor=True)
            embeddings2 = model.encode(description, convert_to_tensor=True)
            score = util.pytorch_cos_sim(embeddings1, embeddings2).item() * 100
            if score < 0:
                score = 0
            cv_name = os.path.splitext(os.path.basename(file.name))[0]
            output = f"{cv_name}\nSimilarity score: {score:.2f}%\nSummary: {summary}\n"
            cv_info.append((output, score))  # Store output and score

        except Exception as e:
            outputs.append(f"Error processing {file.name}: {str(e)}")

    # Sort the CVs by similarity score in descending order
    cv_info.sort(key=lambda x: x[1], reverse=True)
    sorted_outputs = [cv[0] for cv in cv_info]

    outputs.extend(sorted_outputs)

    # Add the total number of PDFs uploaded to the output
    outputs.append(f"Total PDFs uploaded: {total_pdfs}")

    return "\n---\n".join(outputs)


iface = gr.Interface(
    read_pdf,
    [
        gr.inputs.File(file_count="multiple", label="Upload CV in PDF or Word"),
        gr.inputs.Textbox(label="Description", placeholder="Enter job description or keywords with , sepration for checking similarty"),
    ],
    gr.outputs.Textbox(label="Output"),
    title="<span style='font-family: Tahoma; font-weight: bold; color: white; font-size: 48px;'>CDS</span>",
    theme="dark",
    layout="vertical",
    flagging_options=["Incorrect output", "Poor summary", "Low similarity score"],
    flagging_callback=gr.CSVLogger(),
)
# C:\Users\Ammar\Downloads\cv_project\cv_project
# C:\Users\CDS\Desktop\cv_project\cv_project
model_directory = 'C:\\Users\\Ammar\\Downloads\\cv_project\\cv_project'
model_filename = os.path.join(model_directory, 'my_model.pth')

# Load the model from the saved file if it exists; otherwise, create and save the model
if os.path.exists(model_filename):
    model = torch.load(model_filename)
else:
    # Create and load the model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Save the model to the specified directory
    os.makedirs(model_directory, exist_ok=True)  # Create the directory if it doesn't exist
    torch.save(model, model_filename)

# Launch the interface with the share option

url_to_open = "http://127.0.0.1:7860"

# Open the web browser with the specified URL
webbrowser.open(url_to_open)
iface.launch()


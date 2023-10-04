# import os
# import gradio as gr
# from pdfminer.high_level import extract_text
# from transformers import pipeline
# import nltk
# from nltk.corpus import stopwords
# from sentence_transformers import SentenceTransformer, util

# # Download NLTK stopwords if not already downloaded
# nltk.download('stopwords')

# # Load a pre-trained model from the Hugging Face Hub
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# # Create a summarizer pipeline using the sshleifer/distilbart-cnn-12-6 model
# summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# # Define a function for text preprocessing
# def preprocess_text(text):
#     # Tokenize the text
#     tokens = nltk.word_tokenize(text)

#     # Remove stopwords and non-alphabetic tokens
#     stop_words = set(stopwords.words('english'))
#     tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]

#     # Join the tokens back into a single string
#     preprocessed_text = ' '.join(tokens)

#     return preprocessed_text

# # Define a function that takes a list of files and a description as inputs and returns a list of outputs
# def read_pdf(files, description):
#     outputs = []
#     cv_info = []  # Store CV information along with similarity scores

#     for file in files:
#         try:
#             # Read the text from the PDF file
#             text = extract_text(file.name)

#             if not text:
#                 raise Exception("PDF extraction failed or returned empty text.")

#             # Limit the input text length
#             max_input_length = 1000  # Define a maximum length
#             if len(text) > max_input_length:
#                 text = text[:max_input_length]

#             # Preprocess the text
#             preprocessed_text = preprocess_text(text)

#             # Rest of the code remains the same
#             summary = summarizer(preprocessed_text, min_length=50, max_length=1000)[0]["summary_text"]
#             embeddings1 = model.encode(preprocessed_text, convert_to_tensor=True)
#             embeddings2 = model.encode(description, convert_to_tensor=True)
#             score = util.pytorch_cos_sim(embeddings1, embeddings2).item() * 100
#             cv_name = os.path.splitext(os.path.basename(file.name))[0]
#             output = f"{cv_name}\nSimilarity score: {score:.2f}%\nSummary: {summary}\n"
#             cv_info.append((output, score))  # Store output and score

#         except Exception as e:
#             outputs.append(f"Error processing {file.name}: {str(e)}")

#     # Sort the CVs by similarity score in descending order
#     cv_info.sort(key=lambda x: x[1], reverse=True)
#     sorted_outputs = [cv[0] for cv in cv_info]

#     outputs.extend(sorted_outputs)
#     return "\n---\n".join(outputs)

# # Create a Gradio interface with custom logo and CSS
# iface = gr.Interface(
#     read_pdf,
#     [
#         gr.inputs.File(file_count="multiple", label="Upload CV PDFs"),
#         gr.inputs.Textbox(label="Description", placeholder="Enter job description here"),
#     ],
#     gr.outputs.Textbox(label="Output"),
#     title="<span style='font-family: Tahoma; font-weight: bold; color: white; font-size: 48px;'>CDS</span>",
#     theme="dark",
#     layout="vertical",
#     flagging_options=["Incorrect output", "Poor summary", "Low similarity score"],
#     flagging_callback=gr.CSVLogger(),
# )

# # Launch the interface with the share option
# iface.launch(share= True)
import os
import gradio as gr
from pdfminer.high_level import extract_text
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util


# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# Load a pre-trained model from the Hugging Face Hub
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Create a summarizer pipeline using the sshleifer/distilbart-cnn-12-6 model
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

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
            # Read the text from the PDF file
            text = extract_text(file.name)

            if not text:
                raise Exception("PDF extraction failed or returned empty text.")

            # Limit the input text length
            max_input_length = 1000  # Define a maximum length
            if len(text) > max_input_length:
                text = text[:max_input_length]

            # Preprocess the text
            preprocessed_text = preprocess_text(text)

            # Rest of the code remains the same
            summary = summarizer(preprocessed_text, min_length=50, max_length=1000)[0]["summary_text"]
            embeddings1 = model.encode(preprocessed_text, convert_to_tensor=True)
            embeddings2 = model.encode(description, convert_to_tensor=True)
            score = util.pytorch_cos_sim(embeddings1, embeddings2).item() * 100
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
    outputs.append(f"Total CV's uploaded: {total_pdfs}")

    return "\n---\n".join(outputs)

# Create a Gradio interface with custom logo and CSS
iface = gr.Interface(
    read_pdf,
    [
        gr.inputs.File(file_count="multiple", label="Upload CV PDFs"),
        gr.inputs.Textbox(label="Description", placeholder="Enter job description here"),
    ],
    gr.outputs.Textbox(label="Output"),
    title="<span style='font-family: Tahoma; font-weight: bold; color: white; font-size: 48px;'>CDS</span>",
    theme="dark",
    layout="vertical",
    flagging_options=["Incorrect output", "Poor summary", "Low similarity score"],
    flagging_callback=gr.CSVLogger(),
)

# Launch the interface with the share option
iface.launch(share=True)

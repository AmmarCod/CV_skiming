# import os
# from sentence_transformers import SentenceTransformer, util
# import gradio as gr
# from pdfminer.high_level import extract_text
# from transformers import pipeline  # Import the pipeline function

# # Load a pre-trained model from the Hugging Face Hub
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# # Create a summarizer pipeline using the sshleifer/distilbart-cnn-12-6 model
# summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# # Define a function that takes a list of files and a description as inputs and returns a list of outputs
# def read_pdf(files, description):
#     outputs = []
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

#             # Rest of the code remains the same
#             summary = summarizer(text, min_length=50, max_length=1000)[0]["summary_text"]
#             embeddings1 = model.encode(text, convert_to_tensor=True)
#             embeddings2 = model.encode(description, convert_to_tensor=True)
#             score = util.pytorch_cos_sim(embeddings1, embeddings2).item() * 100
#             cv_name = os.path.splitext(os.path.basename(file.name))[0]
#             output = f"{cv_name}\nSimilarity score: {score:.2f}%\nSummary: {summary}\n"
#             outputs.append(output)
#         except Exception as e:
#             outputs.append(f"Error processing {file.name}: {str(e)}")

#     return "\n---\n".join(outputs)

# css_code = “”" .gradio-title { display: flex; align-items: center; justify-content: center; font-size: 36px; font-weight: bold; color: white; background-color: #4CAF50; }

# .gradio-title img { width: 50px; height: 50px; margin-right: 10px; } “”"

# # Create a Gradio interface with file and textbox inputs and textbox outputs
# iface = gr.Interface(
    
#     read_pdf,
#     [gr.inputs.File(file_count="multiple", label="Upload CV PDFs"), gr.inputs.Textbox(label="Description")],
#     gr.outputs.Textbox(label="Output"),
#     title="<img src='https://tse4.mm.bing.net/th?id=OIP.xzNR4Yxj5UkBMjwGlR6aNQAAAA&pid=Api&P=0&h=180'>CV's Checker",  # Use markdown option
#     theme="dark",
#     layout="vertical",  # Use vertical layout to save space
#     flagging_options=["Incorrect output", "Poor summary", "Low similarity score"],  # Add flagging options for feedback
#     flagging_callback=gr.CSVLogger() 
#     css=css_code   

# )


# # Launch the interface with the share option
# iface.launch(share=True)

# import os
# from sentence_transformers import SentenceTransformer, util
# import gradio as gr
# from pdfminer.high_level import extract_text
# from transformers import pipeline  # Import the pipeline function

# # Load a pre-trained model from the Hugging Face Hub
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# # Create a summarizer pipeline using the sshleifer/distilbart-cnn-12-6 model
# summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# # Define a function that takes a list of files and a description as inputs and returns a list of outputs
# def read_pdf(files, description):
#     outputs = []
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

#             # Rest of the code remains the same
#             summary = summarizer(text, min_length=50, max_length=1000)[0]["summary_text"]
#             embeddings1 = model.encode(text, convert_to_tensor=True)
#             embeddings2 = model.encode(description, convert_to_tensor=True)
#             score = util.pytorch_cos_sim(embeddings1, embeddings2).item() * 100
#             cv_name = os.path.splitext(os.path.basename(file.name))[0]
#             output = f"{cv_name}\nSimilarity score: {score:.2f}%\nSummary: {summary}\n"
#             outputs.append(output)
#         except Exception as e:
#             outputs.append(f"Error processing {file.name}: {str(e)}")

#     return "\n---\n".join(outputs)

# # Define CSS styling
# with open('style.css', 'w') as css_file:
#     css_file.write("""
#     .gradio-title {
#         display: flex;
#         align-items: right;
#         justify-content: center;
#         font-size: 100px;
#         font-weight: bold;
#         color: white;
#         background-color: #4CAF50;
#     }

#     .gradio-title img {
#         width: 20px;
#         height: 20px;
#         margin-right: 10px;
#     }
#     """)

# # Create a Gradio interface with file and textbox inputs and textbox outputs
# iface = gr.Interface(
#     read_pdf,
#     [
#         gr.inputs.File(file_count="multiple", label="Upload CV PDFs"),
#         gr.inputs.Textbox(label="Description", placeholder="Enter job description here"),
#     ],
#     gr.outputs.Textbox(label="Output"),
#     title="<img src='https://tse4.mm.bing.net/th?id=OIP.xzNR4Yxj5UkBMjwGlR6aNQAAAA&pid=Api&P=0&h=180'>CV Checker",  # Change the title
#     theme="dark",
#     layout="vertical",
#     flagging_options=["Incorrect output", "Poor summary", "Low similarity score"],
#     flagging_callback=gr.CSVLogger(),
#     css="style.css",  # Apply the CSS styling
# )

# # Launch the interface with the share option
# iface.launch()

# having fun with gui
import os
from sentence_transformers import SentenceTransformer, util
import gradio as gr
from pdfminer.high_level import extract_text
from transformers import pipeline  # Import the pipeline function

# Load a pre-trained model from the Hugging Face Hub
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Create a summarizer pipeline using the sshleifer/distilbart-cnn-12-6 model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Define a function that takes a list of files and a description as inputs and returns a list of outputs
def read_pdf(files, description):
    outputs = []
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

            # Rest of the code remains the same
            summary = summarizer(text, min_length=50, max_length=1000)[0]["summary_text"]
            embeddings1 = model.encode(text, convert_to_tensor=True)
            embeddings2 = model.encode(description, convert_to_tensor=True)
            score = util.pytorch_cos_sim(embeddings1, embeddings2).item() * 100
            cv_name = os.path.splitext(os.path.basename(file.name))[0]
            output = f"{cv_name}\nSimilarity score: {score:.2f}%\nSummary: {summary}\n"
            outputs.append(output)
        except Exception as e:
            outputs.append(f"Error processing {file.name}: {str(e)}")

    return "\n---\n".join(outputs)



# Create a Gradio interface with custom logo and CSS
iface = gr.Interface(
    read_pdf,
    [
        gr.inputs.File(file_count="multiple", label="Upload CV PDFs"),
        gr.inputs.Textbox(label="Description", placeholder="Enter job description here"),
    ],
    gr.outputs.Textbox(label="Output"),
    # title="<span style='font-family:  Cooper Hewitt; color: white; font-size: 48px;'>CDS</span>",
    title="<span style='font-family: Tahoma; font-weight: bold; color: white; font-size: 48px;'>CDS</span>",
    theme="dark",
    layout="vertical",
    flagging_options=["Incorrect output", "Poor summary", "Low similarity score"],
    flagging_callback=gr.CSVLogger(),
  
)

# Launch the interface with the share option
iface.launch()


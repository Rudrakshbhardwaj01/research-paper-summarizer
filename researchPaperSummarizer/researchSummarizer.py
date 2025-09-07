from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import load_prompt
from pathlib import Path
import os

load_dotenv()  # loads local .env
HF_TOKEN = os.getenv("HF_TOKEN")  # Hugging Face API token

# ---------------------------
# Configure HuggingFace LLM
# ---------------------------
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN
)
model = ChatHuggingFace(llm=llm)

st.title("Research Paper Summarizer")

# ---------------------------
# Paper Selection
# ---------------------------
paper_input = st.selectbox("Select Research Paper", [
    "Attention Is All You Need",
    "BERT: Pre-training of Deep Bidirectional Transformers",
    "GPT-3: Language Models are Few-Shot Learners",
    "Diffusion Models Beat GANs on Image Synthesis",
    "GPT-4 Technical Report",
    "XLNet: Generalized Autoregressive Pretraining for Language Understanding",
    "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
    "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations",
    "T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
    "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators",
    "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity",
    "Playing Atari with Deep Reinforcement Learning",
    "Human-level Control through Deep Reinforcement Learning",
    "Mastering the Game of Go with Deep Neural Networks and Tree Search",
    "Mastering the Game of Go without Human Knowledge",
    "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm",
    "Proximal Policy Optimization Algorithms",
    "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor",
    "Continuous Control with Deep Reinforcement Learning (DDPG)",
    "Asynchronous Methods for Deep Reinforcement Learning",
    "MuZero: Mastering Atari, Go, Chess and Shogi without Rules",
    "ImageNet Classification with Deep Convolutional Neural Networks",
    "Very Deep Convolutional Networks for Large-Scale Image Recognition",
    "Deep Residual Learning for Image Recognition",
    "Going Deeper with Convolutions",
    "SqueezeNet: AlexNet-level Accuracy with 50x Fewer Parameters",
    "Densely Connected Convolutional Networks",
    "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks",
    "You Only Look Once: Unified, Real-Time Object Detection",
    "Mask R-CNN",
    "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
    "End-to-End Object Detection with Transformers",
    "Generative Adversarial Networks",
    "Conditional Generative Adversarial Nets",
    "Unsupervised Representation Learning with Deep Convolutional GANs",
    "Wasserstein GAN",
    "Improved Training of Wasserstein GANs",
    "A Style-Based Generator Architecture for Generative Adversarial Networks (StyleGAN)",
    "Analyzing and Improving the Image Quality of StyleGAN (StyleGAN2)",
    "Denoising Diffusion Probabilistic Models",
    "Improved Denoising Diffusion Probabilistic Models",
    "Score-Based Generative Modeling through Stochastic Differential Equations",
    "Imagen: Photorealistic Text-to-Image Diffusion Models",
    "Zero-Shot Text-to-Image Generation (DALL·E)",
    "High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)",
    "Batch Normalization: Accelerating Deep Network Training",
    "Adam: A Method for Stochastic Optimization",
    "Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
    "Pretext-Invariant Representation Learning",
    "A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)",
    "Momentum Contrast for Unsupervised Visual Representation Learning (MoCo)",
    "Bootstrap Your Own Latent (BYOL)",
    "Masked Autoencoders Are Scalable Vision Learners",
    "Learning Transferable Visual Models From Natural Language Supervision (CLIP)",
    "ALIGN: Scaling Up Visual and Vision-Language Representation Learning",
    "BLIP: Bootstrapping Language-Image Pre-training",
    "PaLM: Scaling Language Models",
    "Training Compute-Optimal Large Language Models (Chinchilla)",
    "LLaMA: Open and Efficient Foundation Language Models",
    "Segment Anything",
    "Whisper: Robust Speech Recognition via Large-Scale Weak Supervision"
])


style_input = st.selectbox("Explanation Style", [
    "Beginner-Friendly",
    "Code-Heavy",
    "Mathematically Intuitive",
    "Advanced"
])

length_input = st.selectbox("Explanation Length", [
    "Short (1-2 paragraphs)",
    "Medium (3-5 paragraphs)",
    "Long (detailed explanation)"
])


BASE_DIR = Path(__file__).parent
template_path = BASE_DIR / "template.json"
template = load_prompt(template_path)

if st.button("Summarize"):
    with st.spinner("Generating summary..."):
        chain = template | model
        result = chain.invoke({
            'paper_input': paper_input,
            'style_input': style_input,
            'length_input': length_input
        })
    
    st.subheader("Summary:")
    st.write(result.content)

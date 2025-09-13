#!/usr/bin/env python3
"""
Example usage of the RAG class.

This script demonstrates how to use the RAG class to find the most relevant prompt
for a given query.
"""

from rag import RAG
import os


def main():
    # Set your OpenAI API key (you can also set it as an environment variable)
    # os.environ['OPENAI_API_KEY'] = 'your-api-key-here'
    
    # Example queries
    queries = [
        "how much money do i put in the stock market",
    ]
    
    # Example prompts list (you can load this from your CSV file)
    prompts_list = [
        "What is the step by step guide to invest in share market in india?",
        "What is the story of Kohinoor (Koh-i-Noor) Diamond?",
        "How can I increase the speed of my internet connection while using a VPN?",
        "Why am I mentally very lonely? How can I solve it?",
        "What are the best programming languages to learn in 2024?",
        "How to cook pasta?",
        "What are the benefits of exercise?",
        "How to learn programming?",
        "What is machine learning and how does it work?",
        "How to start a successful business?",
        "What are the health benefits of meditation?",
        "How to improve your communication skills?",
        "What is artificial intelligence?",
        "How to manage stress effectively?",
        "What are the best investment strategies?",
        "How to learn a new language quickly?",
        "What is blockchain technology?",
        "How to build good habits?",
        "What are the benefits of reading books?",
        "How to become a better leader?"
    ]
    
    print("RAG System Example")
    print("=" * 50)
    
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Query: '{query}'")
        print("-" * 30)
        
        try:
            # Initialize RAG system with OpenAI embeddings (paid, faster/better)
            rag = RAG(
                query=query,
                prompts_list=prompts_list,
                top_k=5,  # Retrieve top 5 similar prompts
                embedding_type="openai"  # OpenAI embeddings
            )
            
            # Run the RAG pipeline
            selected_prompt = rag.run()
            
            print(f"Selected prompt: '{selected_prompt}'")
            
        except Exception as e:
            print(f"Error: {e}")
        
        print()


if __name__ == "__main__":
    main()
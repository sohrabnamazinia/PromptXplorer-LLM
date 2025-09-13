#!/usr/bin/env python3
"""
Test script to compare both embedding types
"""

from rag import RAG
import time

def test_both_embeddings():
    """Test both sentence transformer and OpenAI embeddings"""
    
    # Example prompts list
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
    
    query = "How to invest in stocks?"
    
    print("Comparing Embedding Types")
    print("=" * 50)
    print(f"Query: '{query}'")
    print(f"Total prompts: {len(prompts_list)}")
    print()
    
    # Test Sentence Transformer (Free)
    print("1. Testing Sentence Transformer Embeddings (FREE)")
    print("-" * 50)
    start_time = time.time()
    
    try:
        rag_st = RAG(
            query=query,
            prompts_list=prompts_list,
            top_k=5,
            embedding_type="sentence_transformer"
        )
        
        similar_prompts_st = rag_st._retrieve_similar_prompts()
        st_time = time.time() - start_time
        
        print("Top 5 similar prompts:")
        for i, prompt in enumerate(similar_prompts_st, 1):
            print(f"{i}. {prompt}")
        
        print(f"\nTime taken: {st_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error with sentence transformer: {e}")
    
    print("\n" + "="*50)
    
    # Test OpenAI Embeddings (Paid)
    print("2. Testing OpenAI Embeddings (PAID)")
    print("-" * 50)
    start_time = time.time()
    
    try:
        rag_openai = RAG(
            query=query,
            prompts_list=prompts_list,
            top_k=5,
            embedding_type="openai"
        )
        
        similar_prompts_openai = rag_openai._retrieve_similar_prompts()
        openai_time = time.time() - start_time
        
        print("Top 5 similar prompts:")
        for i, prompt in enumerate(similar_prompts_openai, 1):
            print(f"{i}. {prompt}")
        
        print(f"\nTime taken: {openai_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error with OpenAI embeddings: {e}")
    
    print("\n" + "="*50)
    print("Comparison Summary:")
    print("- Sentence Transformer: FREE, local processing")
    print("- OpenAI Embeddings: PAID (~$0.0001 per 1K tokens), potentially faster/better quality")
    print("- Both use the same retrieval logic and LLM selection")

if __name__ == "__main__":
    test_both_embeddings()
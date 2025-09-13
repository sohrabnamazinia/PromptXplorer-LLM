#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) Class

This class implements a RAG system that:
1. Takes a query and loads a dataset of prompts
2. Uses FAISS for similarity-based retrieval of top-k prompts
3. Uses an LLM to select exactly one prompt from the retrieved candidates
4. Returns the selected prompt

Usage:
    rag = RAG(query="How to invest in stocks?", dataset_path="data/prompts.csv")
    selected_prompt = rag.run()
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os
from pathlib import Path
from typing import List, Optional


class RAG:
    def __init__(self, query: str, prompts_list: List[str], top_k: int = 5, embedding_type: str = "sentence_transformer"):
        """
        Initialize RAG system.
        
        Args:
            query (str): The input query to find similar prompts for
            prompts_list (List[str]): List of prompts to search through
            top_k (int): Number of top similar prompts to retrieve (default: 5)
            embedding_type (str): Type of embeddings to use - "sentence_transformer" (free) or "openai" (paid, faster/better)
        """
        self.query = query
        self.prompts_list = prompts_list
        self.top_k = top_k
        self.embedding_type = embedding_type
        self.model_name = "all-MiniLM-L6-v2"  # Fixed reasonable model for sentence transformers
        
        # Initialize components
        self.sentence_model = None
        self.prompt_embeddings = None
        self.openai_client = None
        
        # Initialize embeddings and OpenAI client
        self._initialize_openai()
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize embeddings based on the selected type."""
        try:
            if self.embedding_type == "sentence_transformer":
                # Initialize sentence transformer
                self.sentence_model = SentenceTransformer(self.model_name)
                
                # Encode all prompts
                print("Encoding prompts with sentence transformer...")
                self.prompt_embeddings = self.sentence_model.encode(self.prompts_list)
                
            elif self.embedding_type == "openai":
                # Use OpenAI embeddings
                print("Encoding prompts with OpenAI embeddings...")
                self.prompt_embeddings = self._get_openai_embeddings(self.prompts_list)
                
            else:
                raise ValueError(f"Unsupported embedding_type: {self.embedding_type}. Use 'sentence_transformer' or 'openai'")
            
            print(f"Computed embeddings for {len(self.prompts_list)} prompts using {self.embedding_type}")
            
        except Exception as e:
            raise Exception(f"Error initializing embeddings: {e}")
    
    def _get_openai_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get OpenAI embeddings for a list of texts."""
        try:
            embeddings = []
            # Process in batches to avoid rate limits
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = self.openai_client.embeddings.create(
                    model="text-embedding-3-small",  # Fast and cost-effective
                    input=batch
                )
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
            
            return np.array(embeddings)
            
        except Exception as e:
            raise Exception(f"Error getting OpenAI embeddings: {e}")
    
    def _initialize_openai(self):
        """Initialize OpenAI client."""
        try:
            # Check for OpenAI API key
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise Exception("OPENAI_API_KEY environment variable not set")
            
            self.openai_client = OpenAI(api_key=api_key)
            print("OpenAI client initialized")
            
        except Exception as e:
            raise Exception(f"Error initializing OpenAI: {e}")
    
    def _retrieve_similar_prompts(self) -> List[str]:
        """
        Retrieve top-k most similar prompts to the query.
        
        Returns:
            List[str]: List of top-k similar prompts
        """
        try:
            # Encode query based on embedding type
            if self.embedding_type == "sentence_transformer":
                query_embedding = self.sentence_model.encode([self.query])
            elif self.embedding_type == "openai":
                query_embedding = self._get_openai_embeddings([self.query])
            else:
                raise ValueError(f"Unsupported embedding_type: {self.embedding_type}")
            
            # Compute cosine similarities
            similarities = cosine_similarity(query_embedding, self.prompt_embeddings)[0]
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[-self.top_k:][::-1]  # Sort in descending order
            
            # Get the actual prompts
            similar_prompts = [self.prompts_list[idx] for idx in top_indices]
            
            print(f"Retrieved {len(similar_prompts)} similar prompts using {self.embedding_type}")
            return similar_prompts
            
        except Exception as e:
            raise Exception(f"Error retrieving similar prompts: {e}")
    
    def _select_prompt_with_llm(self, candidate_prompts: List[str]) -> str:
        """
        Use LLM to select exactly one prompt from candidates.
        
        Args:
            candidate_prompts (List[str]): List of candidate prompts
            
        Returns:
            str: The selected prompt
        """
        try:
            # Create system prompt
            system_prompt = """You are a prompt selection expert. Your task is to select exactly ONE prompt from the given candidates that best matches the user's query.

Rules:
1. You must select exactly ONE prompt
2. Return ONLY the selected prompt text, nothing else
3. No explanations, no additional words
4. The prompt should be most relevant to the user's query

Candidate prompts:"""
            
            # Create user message with candidates
            candidates_text = "\n".join([f"{i+1}. {prompt}" for i, prompt in enumerate(candidate_prompts)])
            user_message = f"{system_prompt}\n\n{candidates_text}\n\nUser Query: {self.query}\n\nSelected prompt:"
            
            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {self.query}\n\nCandidates:\n{candidates_text}\n\nSelect exactly one prompt:"}
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            selected_prompt = response.choices[0].message.content.strip()
            
            # Validate that the selected prompt is one of the candidates
            if selected_prompt not in candidate_prompts:
                # Try to find the closest match
                for candidate in candidate_prompts:
                    if candidate.strip() in selected_prompt or selected_prompt in candidate.strip():
                        selected_prompt = candidate
                        break
                else:
                    # If no match found, return the first candidate
                    selected_prompt = candidate_prompts[0]
                    print("Warning: LLM response didn't match any candidate, using first candidate")
            
            print(f"LLM selected prompt: {selected_prompt[:100]}...")
            return selected_prompt
            
        except Exception as e:
            raise Exception(f"Error selecting prompt with LLM: {e}")
    
    def run(self) -> str:
        """
        Run the complete RAG pipeline.
        
        Returns:
            str: The selected prompt
        """
        try:
            print(f"Running RAG pipeline for query: '{self.query}'")
            
            # Step 1: Retrieve similar prompts
            similar_prompts = self._retrieve_similar_prompts()
            
            # Step 2: Use LLM to select one prompt
            selected_prompt = self._select_prompt_with_llm(similar_prompts)
            
            print("RAG pipeline completed successfully")
            return selected_prompt
            
        except Exception as e:
            raise Exception(f"Error in RAG pipeline: {e}")


def main():
    """Example usage of the RAG class."""
    # Example usage
    query = "How to invest in stocks?"
    
    # Example prompts list
    prompts_list = [
        "What is the step by step guide to invest in share market in india?",
        "What is the story of Kohinoor (Koh-i-Noor) Diamond?",
        "How can I increase the speed of my internet connection while using a VPN?",
        "Why am I mentally very lonely? How can I solve it?",
        "What are the best programming languages to learn in 2024?"
    ]
    
    try:
        # Example with sentence transformer (free)
        print("Testing with sentence transformer embeddings (free):")
        rag = RAG(query=query, prompts_list=prompts_list, top_k=5, embedding_type="sentence_transformer")
        selected_prompt = rag.run()
        
        print(f"\nFinal selected prompt:")
        print(f"'{selected_prompt}'")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
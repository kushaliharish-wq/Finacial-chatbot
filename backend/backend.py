import os
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import google.generativeai as genai  # Assuming genai is available for Google Gemini

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
from datetime import datetime
import uvicorn


# Initialize OpenAI client
api_key = "OPEN_AI_KEY"  # Replace with your OpenAI API Key
client = OpenAI(api_key=api_key)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Pinecone client
pc = Pinecone(api_key="PINECONE_API_KEY")
index_name = "final2-index"
index = pc.Index(index_name)

# Initialize Google Gemini (genai) API client
genai.configure(api_key="GENAI_KEY")
model = genai.GenerativeModel("gemini-pro")


def generate_enhanced_response(transaction_data: str, user_question: str) -> str:
    """Generate a contextual response combining transaction data with insights."""

    # First check if we got any transaction data
    if not transaction_data or "No transaction found" in transaction_data:
        return "I couldn't find any matching transactions for your query. Please check the date and category and try again."

    prompt = f"""
    User Question: {user_question}
    Transaction Data: {transaction_data}
    
    Rules:
    1. If transaction data shows specific transaction details, format response as:
       "On [date], you spent $[amount] for [category] ([description])"
    2. If transaction data shows aggregate spending, format response as:
       "In [date], you spent a total of $[amount] on [category] across [number] transactions"
    3. Always include the exact amounts from the transaction data
    4. Be precise and direct in answering the amount spent
    
    Please provide a clear, direct response focusing on the amount spent.
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating response: {e}")
        # Fall back to direct transaction data
        return transaction_data


# Function to extract entities from query
def extract_entities_from_query(query):
    prompt = f"""
    User query: "{query}"
    
    Extract the following details:
    - transaction_category: Category of spending (e.g., shopping, groceries,alcohol & bars,auto insurance,coffee shops,electronics & software, entertainment,fast food,gas & fuel,haircut,home improvement, internet, mobile phone,mortgage & rent,movies & dvds,	
music,restaurants,shopping,television,utilities)
    - description: Store or vendor name (like Thai Restaurant, Amazon, Netflix,Credit Card Payment,Mortgage Payment,Hardware Store,Phone Company,Grocery Store,	
City Water Charges,Biweekly Paycheck,Internet Service Provider,Brunch Restaurant,Barbershop,Brewing Company,BP,	
Movie Theater)
    - date_range: Specific date or range (YYYY-MM-DD or YYYY-MM)
    - amount: If the query asks for a specific amount

    Rules for EXACT category matching:
    1. If the word "internet" appears in the query, ALWAYS use "internet" as the category
    2. Categories must match EXACTLY one of these (do not substitute or combine):
       - shopping
       - groceries
       - alcohol & bars
       - auto insurance
       - coffee shops
       - electronics & software
       - entertainment
       - fast food
       - gas & fuel
       - haircut
       - home improvement
       - internet
       - mobile phone
       - mortgage & rent
       - movies & dvds
       - music
       - restaurants
       - utilities
    
    Rules for dates:
    2. For exact date queries (e.g., "2018-03-26"), set query_type as "single transaction"
    3. For month queries (e.g., "March 2018"), set query_type as "aggregate"
    4. transaction_category must be lowercase and one of: shopping, groceries, restaurants, utilities
    5. query_type must be exactly "single" or "aggregate"

    Format as JSON:
    {{
        "transaction_category": "string",
        "description": "string",
        "date_range": "string",
        "amount": "float",
        "query_type": "string"
    }}
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a financial assistant."},
                {"role": "user", "content": prompt},
            ],
            model="gpt-3.5-turbo",
        )

        response_content = chat_completion.choices[0].message.content

        if response_content.startswith("```json"):
            response_content = response_content[7:-3].strip()
        elif response_content.startswith("```"):
            response_content = response_content[3:-3].strip()

        print(f"Raw response content: {response_content}")  # Debug print

        entities = json.loads(response_content)
        print(f"Extracted entities: {entities}")
        return entities

    except json.JSONDecodeError as e:
        print(f"Error parsing response as JSON: {e}")
        print(f"Response content was: {response_content}")
        return {}


# Function to search Pinecone dynamically based on user query type
def search_pinecone(query):
    entities = extract_entities_from_query(query)
    query_type = entities.get("query_type", "")
    # Fix date extraction - check both 'date' and 'date_range' fields
    date = entities.get("date_range", entities.get("date", ""))
    print(f"Extracted date: {date}")  # Debug print
    if query_type == "single transaction":
        # Handle single transaction search
        category = entities.get("transaction_category", "").lower()
        print(f"Searching for category: {category}")  # Debug print

        # Create embeddings with category instead of description
        category_embedding = embed_model.encode(f"category: {category}")
        date_embedding = embed_model.encode(f"date: {date}")
        query_embedding = concatenate_embeddings(date_embedding, category_embedding)

        search_results = index.query(
            vector=query_embedding.tolist(),
            top_k=5,  # Get a few results to find best match
            include_metadata=True,
        )

        return format_single_transaction(search_results, category, date)

    elif query_type == "aggregate":
        category = entities.get("transaction_category", "").lower()
        print(f"Extracted category: {category}")  # Debug print

        if not date:
            return "Please specify a date in YYYY-MM format."

        # Ensure date is in YYYY-MM format
        date = date[:7] if len(date) > 7 else date

        category_embedding = embed_model.encode(f"category: {category}")
        date_embedding = embed_model.encode(f"date: {date}")
        query_embedding = concatenate_embeddings(date_embedding, category_embedding)

        search_results = index.query(
            vector=query_embedding.tolist(), top_k=100, include_metadata=True
        )

        return format_aggregate_transactions(search_results, date, category)

    return "Could not determine the query type. Please try again."


# Concatenate embeddings (padding and truncation)
def concatenate_embeddings(*embeddings, target_dim=1920):
    """Combine embeddings with proper dimensionality."""
    combined = []
    for emb in embeddings:
        combined.extend(emb)

    if len(combined) < target_dim:
        combined.extend([0] * (target_dim - len(combined)))
    return np.array(combined[:target_dim], dtype=np.float64)


# Format aggregate transactions output
def format_single_transaction(results, category, date):
    """Format results for a single transaction with exact date matching."""
    print(f"\nSearching for {category} transaction on {date}")

    if not results["matches"]:
        return "No matching transaction found."

    print("\nExamining matches:")
    matched_transactions = []

    for match in results["matches"]:
        metadata = match["metadata"]
        print(f"\nChecking transaction:")
        print(f"Date: {metadata['date']}")
        print(f"Category: {metadata['transaction_category'].lower()}")
        print(f"Amount: ${float(metadata['amount']):.2f}")
        print(f"Score: {match['score']}")
        exact_date_match = metadata["date"] == date
        category_match = metadata["transaction_category"].lower() == category
        # For single transactions, match exact date and category
        print(f"Date match: {exact_date_match}")
        print(f"Category match: {category_match}")

        if exact_date_match and category_match:
            print("✓ MATCH")
            matched_transactions.append((match["score"], metadata))
        else:
            print("✗ NO MATCH")
            if not exact_date_match:
                print(
                    f"  Reason: Date mismatch (expected {date}, got {metadata['date']})"
                )
            if not category_match:
                print(
                    f"  Reason: Category mismatch (expected {category}, got {metadata['transaction_category'].lower()})"
                )

    if not matched_transactions:
        return f"No transaction found for {category} on {date}."

    # Get the highest scoring match
    best_match = max(matched_transactions, key=lambda x: x[0])
    metadata = best_match[1]

    return f"""
    **Transaction Details:**
    - **Date:** {metadata['date']}
    - **Description:** {metadata['description']}
    - **Amount:** ${float(metadata['amount']):.2f}
    - **Category:** {metadata['transaction_category']}
    - **Account:** {metadata.get('account_name', 'N/A')}
    """


def format_aggregate_transactions(results, date, category):
    """Format aggregate results with precise date matching."""
    print(f"\nProcessing transactions for {category} in {date}")
    print("----------------------------------------")

    # Filter and print matching transactions
    matching_transactions = []
    total_spent = 0.0

    print("\nAll transactions being considered:")
    for match in results["matches"]:
        metadata = match["metadata"]
        transaction_date = metadata["date"][:7]  # Get just YYYY-MM part

        print(f"\nChecking transaction:")
        print(f"Date: {metadata['date']} (YYYY-MM: {transaction_date})")
        print(f"Looking for date: {date}")
        print(f"Category: {metadata['transaction_category'].lower()}")
        print(f"Description: {metadata['description']}")
        print(f"Amount: ${float(metadata['amount']):.2f}")

        # Check if transaction matches criteria exactly
        matches_date = transaction_date == date
        matches_category = metadata["transaction_category"].lower() == category

        if matches_date and matches_category:
            print("✓ MATCH - Transaction included in calculation")
            matching_transactions.append(metadata)
            total_spent += float(metadata["amount"])
        else:
            print("✗ NO MATCH - Transaction excluded")
            if not matches_date:
                print(
                    f"  Reason: Date mismatch (expected {date}, got {transaction_date})"
                )
            if not matches_category:
                print(
                    f"  Reason: Category mismatch (expected {category}, got {metadata['transaction_category'].lower()})"
                )

    if not matching_transactions:
        return f"No transactions found for category '{category}' in {date}."

    # Sort transactions by date
    matching_transactions.sort(key=lambda x: x["date"])

    transactions_list = "\n".join(
        [
            f"- {tx['date']}: {tx['description']} (${float(tx['amount']):.2f})"
            for tx in matching_transactions
        ]
    )

    summary = f"""
    **Spending Summary for {category.title()} in {date}:**
    - **Total Amount:** ${total_spent:.2f}
    - **Number of Transactions:** {len(matching_transactions)}

    **Individual Transactions:**
    {transactions_list}
    """

    print("\nFinal Summary:")
    print(f"Total matching transactions: {len(matching_transactions)}")
    print(f"Total amount: ${total_spent:.2f}")

    return summary


# Initialize FastAPI
app = FastAPI()


# Define the request body schema
class QueryRequest(BaseModel):
    user_id: str
    question: str
    chat_history: list[str]


# Dummy chat history store (in-memory, for demonstration purposes)
chat_history = {}


@app.post("/query")
async def handle_query(request: QueryRequest):
    try:
        # First, search for relevant transaction data
        transaction_data = search_pinecone(request.question)
        print(transaction_data)

        # Generate an enhanced response using the transaction data
        response = generate_enhanced_response(transaction_data, request.question)
        print(f"\nGenerated response: {response}")
        return {"response": response, "transaction_data": transaction_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8001)

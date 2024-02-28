from transformers import BertTokenizer, BertForSequenceClassification, pipeline

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # Assuming binary classification (positive/negative)

def analyze_sentiment(context, text):
    # Combine context and text
    combined_text = f"{context}: {text}"
    
    # Tokenize the combined text
    tokenized_text = tokenizer(combined_text, padding='max_length', truncation=True, return_tensors="pt")
    
    # Perform sentiment analysis
    outputs = model(**tokenized_text)
    logits = outputs.logits
    predicted_class = logits.argmax().item()
    
    # Return sentiment label
    if predicted_class == 0:
        return "Negative"
    elif predicted_class == 1:
        return "Positive"
    else:
        return "Neutral"

# Example usage
context = input("Please enter the context for analysis")
text = input("Enter the sentiment for analysis")
sentiment = analyze_sentiment(context, text)
print(f"Sentiment: {sentiment}")
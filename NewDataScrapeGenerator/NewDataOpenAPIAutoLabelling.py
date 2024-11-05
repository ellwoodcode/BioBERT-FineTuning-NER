import openai
import time

openai.api_key = "OPEN_API_KEY"

# Function to get IOB formatted sentence from ChatGPT
def get_iob_format(sentence):
    prompt = f"Please format the following sentence using IOB tags with the labels: O, and Disease. Every line should be formatted \"Token Label\". Do not return anything except the formatted sentence, no confirmations of task, no explanation of method. One token and label per line.\nSentence: {sentence}"
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return None

# Function to split text into sentences
def split_into_sentences(text):
    return text.split('.')

# Main function to process the text file
def process_text_file(input_file, output_file):
    counter = 0
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()
        sentences = split_into_sentences(text)
    
    print(f"Total number of sentences: {len(sentences)}")
    onePercent = len(sentences)//100
    startTime = time.time()
    with open(output_file, 'a', encoding='utf-8') as file:
        for sentence in sentences:
            if counter % onePercent == 0:
                percentComplete = counter/onePercent
                presentTime = time.time()
                elapsedTime = presentTime-startTime
                estimatedRemainingTime = elapsedTime*(100-percentComplete)
                estimatedHours = estimatedRemainingTime//3600
                estimatedminutes = (estimatedRemainingTime%3600)//60
                estimatedSeconds = (estimatedRemainingTime%3600)%60
                print(f"{percentComplete}% Completed")
                print(f"Estimated time remaining {estimatedHours}:{estimatedminutes}:{estimatedSeconds}")
                startTime=presentTime

            counter +=1
            iob_formatted = get_iob_format(sentence)
            if iob_formatted:
                file.write(iob_formatted + "\n\n")  # Add a blank line after each formatted sentence

if __name__ == "__main__":
    input_file = "input.txt"  # Replace with the path to your input text file
    output_file = "output.txt"  # Replace with the desired path for your output file
    process_text_file(input_file, output_file)
    print(f"Processing complete. IOB formatted sentences are saved in {output_file}")

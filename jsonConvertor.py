import json
import gzip

def extract_conversation_threads(message_node, current_thread, all_threads):
    """
    Recursively travels down the conversation tree to extract flat, linear conversations.
    """
    # Map the OASST1 "prompter" role to "User"
    role_map = {"prompter": "User", "assistant": "Assistant"}
    role = role_map.get(message_node.get('role'), "Unknown")
    
    # Extract text and replace newlines to keep messages on single lines
    text = message_node.get('text', '').replace('\n', ' ') 
    
    current_thread.append(f"{role}: {text}")
    
    replies = message_node.get('replies', [])
    if not replies:
        # End of the branch reached; save the thread
        all_threads.append('\n'.join(current_thread))
    else:
        # Keep digging down each branch
        for reply in replies:
            extract_conversation_threads(reply, list(current_thread), all_threads)

# Point to the downloaded file inside your cloned directory
input_file = 'oasst1/2023-04-12_oasst_ready.trees.jsonl.gz'
output_file = 'conversation_corpus.txt'

print(f"Reading from {input_file}...")

# Use gzip.open with 'rt' (read text) mode to handle the compressed file directly
with gzip.open(input_file, 'rt', encoding='utf-8') as f_in, \
     open(output_file, 'w', encoding='utf-8') as f_out:
     
    for line in f_in:
        tree_data = json.loads(line)
        
        # The root message is under the "prompt" key
        root_message = tree_data['prompt']
        
        all_threads = []
        extract_conversation_threads(root_message, [], all_threads)
        
        # Write each full conversation branch separated by a line
        for thread in all_threads:
            f_out.write(thread + '\n\n---\n\n')

print(f"Extraction complete. Check {output_file}!")
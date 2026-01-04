import re

def extract_array(file_content, array_name, output_file):
    print(f"🔍 Looking for {array_name}...")
    pattern = re.compile(rf"float\s+{array_name}\s*\[.*?\]\s*=\s*\{{(.*?)\}};", re.DOTALL)
    match = pattern.search(file_content)
    
    if match:
        # Clean up the C syntax to get just numbers
        clean_text = match.group(1).replace('f', '').replace('\n', '').replace('\r', '')
        numbers = [x.strip() for x in clean_text.split(',') if x.strip()]
        
        with open(output_file, 'w') as f:
            f.write(','.join(numbers))
        print(f"   ✅ Extracted {len(numbers)} samples to {output_file}")
    else:
        print(f"   ❌ Could not find {array_name}")

try:
    with open('one_window.h', 'r') as f:
        content = f.read()
    extract_array(content, "apnea_window", "apnea_data.txt")
    extract_array(content, "normal_window", "normal_data.txt")
except FileNotFoundError:
    print("❌ Error: Put this script in the same folder as one_window.h")
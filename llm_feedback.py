# llm_feedback.py
import openai
import os # To potentially load API key from environment variables


# --- OpenAI API Configuration ---
# IMPORTANT: For production, load your API key from an environment variable
# or a secure configuration file, NOT directly in the code.
# Example: openai.api_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = "AP_Key" # Replace with your actual OpenAI API key


# Initialize the OpenAI client (for newer OpenAI library versions)
# client = openai.OpenAI(api_key=openai.api_key) # Uncomment if using openai>=1.0.0


def get_llm_feedback(pose_name, posture_issues):
    """
    Generates natural language feedback using the OpenAI API.
    Args:
        pose_name (str): The name of the detected yoga pose.
        posture_issues (list): A list of strings describing detected posture issues.
    Returns:
        str: A natural language feedback message from the LLM.
    """
    prompt_content = ""
    if not posture_issues:
        # If no issues, provide an encouraging spiritual affirmation
        prompt_content = (
            f"The user is doing a perfect {pose_name}. "
            f"Give them a short, encouraging spiritual affirmation related to yoga, mindfulness, or inner peace. "
            f"Keep it concise, around 1-2 sentences."
        )
    else:
        # If issues exist, provide corrective guidance and relate it to spiritual growth
        issues_str = ", ".join(posture_issues)
        prompt_content = (
            f"The user is attempting a {pose_name} and has the following posture issues: {issues_str}. "
            f"Provide concise, helpful, and encouraging corrective guidance. "
            f"Suggest how this adjustment can deepen their yoga practice or connect to inner balance. "
            f"Keep it to 2-3 sentences."
        )
   
    try:
        # Make the API call to OpenAI's chat completions endpoint
        # Using gpt-3.5-turbo as a common and cost-effective model
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo", # Or "gpt-4" if you have access and prefer higher quality
            messages=[
                {"role": "system", "content": "You are a kind, encouraging, and knowledgeable yoga instructor providing posture feedback and spiritual guidance."},
                {"role": "user", "content": prompt_content}
            ],
            max_tokens=100, # Limit response length to keep it concise
            temperature=0.7 # Adjust creativity (0.0 for more deterministic, 1.0 for more creative)
        )
       
        # Extract the content from the response
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content.strip()
        else:
            return "Unable to generate feedback. No response from LLM."


    except openai.APIError as e:
        print(f"OpenAI API Error: {e}")
        return f"Error connecting to AI. Please check your API key and internet connection."
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return "Unable to provide advanced feedback at this moment due to an internal error."


if __name__ == "__main__":
    # Test the LLM feedback function with dummy data
    # IMPORTANT: This test will only work if you replace "YOUR_CHATGPT_API_KEY" with a valid key.
    print("Testing LLM feedback with no issues (requires valid API key):")
    print(get_llm_feedback("Mountain Pose", []))
   
    print("\nTesting LLM feedback with issues (requires valid API key):")
    print(get_llm_feedback("Tree Pose", ["Straighten your standing leg more.", "Open your lifted knee more to the side."]))

# PROMPTS/bill_summary_prompt.py
BILL_SUMMARY_PROMPT = """
You are an expert legislative analyst specializing in AI governance and technology policy. Your task is to provide a clear, concise summary of the given bill text.

Please analyze the bill and provide a comprehensive summary that includes:

1. **Main Purpose**: What is the primary objective of this bill?
2. **Key Provisions**: What are the main requirements, prohibitions, or authorizations?
3. **AI-Related Elements**: How does this bill relate to artificial intelligence, if at all?
4. **Scope and Impact**: Who does this bill affect and what are the potential consequences?
5. **Implementation**: What mechanisms or processes does the bill establish?

**Requirements:**
- Keep the summary concise but comprehensive (aim for 200-400 words)
- Use clear, professional language
- Focus on the most important aspects of the bill
- If the bill is not related to AI, clearly state this
- Structure the response with clear sections using markdown formatting

**Bill Information:**
- Bill Number: {bill_number}
- Bill Title: {bill_title}
- State: {state}

**Bill Text:**
{bill_text}

Please provide your analysis:
""" 
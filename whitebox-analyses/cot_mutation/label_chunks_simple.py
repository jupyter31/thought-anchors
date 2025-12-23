"""Simple script to label chunks with function tags using Azure Foundry or OpenAI."""
import argparse
import json
import sys
from pathlib import Path
from dotenv import load_dotenv
import os

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from prompts import DAG_PROMPT
from clients.azure_foundry_client import AzureFoundryClient

# Make OpenAI import optional
try:
    from openai import OpenAI, AzureOpenAI
except ImportError:
    OpenAI = None
    AzureOpenAI = None


def label_chunks_for_problem(problem_dir: Path, client, model_name: str = "gpt-4o", model_suffix: str = None, force_relabel: bool = False):
    """
    Label all chunks in a problem directory with function tags.
    
    Args:
        problem_dir: Path to problem directory
        client: LLM client (OpenAI or AzureFoundryClient)
        model_name: Model name to use for labeling
        model_suffix: Optional suffix for field names (e.g., "_gpt_4o" -> function_tags_gpt_4o)
        force_relabel: Force relabeling even if chunks_labeled.json exists
    """
    # Check if chunks_labeled.json already exists
    labeled_chunks_file = problem_dir / "chunks_labeled.json"
    
    # Determine field names based on model suffix
    function_tags_field = f"function_tags{model_suffix}" if model_suffix else "function_tags"
    depends_on_field = f"depends_on{model_suffix}" if model_suffix else "depends_on"
    
    # Only skip if no model suffix and already labeled
    if labeled_chunks_file.exists() and not force_relabel and not model_suffix:
        print(f"Skipping {problem_dir.name} - already labeled (use -f to force)")
        return
    
    # Load problem
    problem_file = problem_dir / "problem.json"
    if not problem_file.exists():
        print(f"No problem.json in {problem_dir.name}")
        return
    
    with open(problem_file, 'r', encoding='utf-8') as f:
        problem_data = json.load(f)
    
    problem_text = problem_data.get("prompt", "")
    if not problem_text:
        problem_text = problem_data.get("uid", "")
    
    # Load chunks
    chunks_file = problem_dir / "chunks_labeled.json"
    if chunks_file.exists():
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
    else:
        print(f"No chunks_labeled.json in {problem_dir.name}")
        return
    
    # Extract chunk texts
    chunks = [chunk.get("text", "") for chunk in chunks_data]
    
    if not chunks:
        print(f"No chunks found in {problem_dir.name}")
        return
    
    print(f"\nLabeling {len(chunks)} chunks in {problem_dir.name}...")
    
    # Create full chunked text
    full_chunked_text = ""
    for i, chunk in enumerate(chunks):
        full_chunked_text += f"Chunk {i}:\n{chunk}\n\n"
    
    # Format the DAG prompt
    formatted_prompt = DAG_PROMPT.format(
        problem_text=problem_text,
        full_chunked_text=full_chunked_text
    )
    
    # Call LLM
    try:
        if isinstance(client, AzureFoundryClient):
            json_instruction = "\n\nRespond with valid JSON only."
            result_data = client.send_chat_request(
                model_name=model_name,
                request={
                    "messages": [{"role": "user", "content": formatted_prompt + json_instruction}],
                    "temperature": 0.0,
                }
            )
            labels = json.loads(result_data["text"])
        else:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            labels = json.loads(response.choices[0].message.content)
        
        # Update chunks with labels
        for chunk_data in chunks_data:
            chunk_idx = chunk_data.get("chunk_idx") or chunk_data.get("chunk_id")
            chunk_id_str = str(chunk_idx)
            
            if chunk_id_str in labels:
                label_info = labels[chunk_id_str]
                chunk_data[function_tags_field] = label_info.get("function_tags", [])
                chunk_data[depends_on_field] = label_info.get("depends_on", [])
        
        # Save updated chunks
        with open(labeled_chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        if model_suffix:
            print(f"✓ Labeled {problem_dir.name} with {function_tags_field}")
        else:
            print(f"✓ Labeled {problem_dir.name}")
        
    except Exception as e:
        print(f"✗ Error labeling {problem_dir.name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Label chunks with function tags")
    parser.add_argument(
        "-d", "--problem_dir",
        type=str,
        required=True,
        help="Path to problem directory or parent directory containing problem folders"
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Force relabeling even if chunks already labeled"
    )
    parser.add_argument(
        "--use-azure-foundry",
        action="store_true",
        help="Use Azure Foundry client instead of OpenAI"
    )
    parser.add_argument(
        "--use-azure-openai",
        action="store_true",
        help="Use Azure OpenAI endpoint instead of standard OpenAI"
    )
    parser.add_argument(
        "--azure-api-key",
        type=str,
        default=None,
        help="Azure API key (defaults to AZURE_API_KEY env var)"
    )
    parser.add_argument(
        "--azure-endpoint",
        type=str,
        default="https://model-ft-test.services.ai.azure.com/models",
        help="Azure inference endpoint (default: https://model-ft-test.services.ai.azure.com/models)"
    )
    parser.add_argument(
        "--azure-model",
        type=str,
        default="gpt-oss-120b",
        help="Azure model deployment name (default: gpt-oss-120b)"
    )
    parser.add_argument(
        "--model-suffix",
        type=str,
        default=None,
        help="Suffix for field names (e.g., '_gpt_4o' stores as function_tags_gpt_4o)"
    )
    parser.add_argument(
        "--azure-api-version",
        type=str,
        default="2024-12-01-preview",
        help="Azure OpenAI API version (default: 2024-12-01-preview)"
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Initialize client
    if args.use_azure_foundry:
        print(f"Using Azure Foundry client (endpoint: {args.azure_endpoint}, model: {args.azure_model})")
        client = AzureFoundryClient(
            api_key=args.azure_api_key,
            endpoint=args.azure_endpoint,
            model_name=args.azure_model,
        )
    elif args.use_azure_openai:
        if AzureOpenAI is None:
            raise ImportError("openai package is required for Azure OpenAI. Install with: pip install openai")
        print(f"Using Azure OpenAI endpoint (endpoint: {args.azure_endpoint}, model: {args.azure_model})")
        client = AzureOpenAI(
            api_key=args.azure_api_key or os.getenv("AZURE_API_KEY"),
            api_version=args.azure_api_version,
            azure_endpoint=args.azure_endpoint,
        )
    else:
        if OpenAI is None:
            raise ImportError("openai package is required when not using Azure Foundry. Install with: pip install openai")
        print("Using OpenAI client")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if not client.api_key:
            raise ValueError("OPENAI_API_KEY not found")
    
    # Process directories
    problem_path = Path(args.problem_dir)
    
    if not problem_path.exists():
        print(f"Error: {problem_path} does not exist")
        return
    
    # Check if it's a single problem directory or parent
    if (problem_path / "problem.json").exists():
        # Single problem directory
        label_chunks_for_problem(problem_path, client, args.azure_model, args.model_suffix, args.force)
    else:
        # Parent directory - process all problem_* subdirectories
        problem_dirs = sorted([d for d in problem_path.iterdir() 
                              if d.is_dir() and d.name.startswith("problem_")])
        
        if not problem_dirs:
            print(f"No problem directories found in {problem_path}")
            return
        
        print(f"Found {len(problem_dirs)} problem directories")
        
        for problem_dir in problem_dirs:
            label_chunks_for_problem(problem_dir, client, args.azure_model, args.model_suffix, args.force)
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()

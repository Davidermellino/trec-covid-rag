#!/usr/bin/env python3
"""
Script veloce per testare Ollama e trovare il problema di performance
"""

import time
import requests
import json
from llama_index.llms.ollama import Ollama

def test_ollama_direct():
    """Test diretto con API Ollama"""
    print("🔄 Testing Ollama API directly...")
    
    url = "http://localhost:11434/api/generate"
    
    query = "COVID-19 treatment effectiveness"
    prompt = f"""Write a short scientific passage about COVID-19 research that answers: "{query}"

Write 2-3 sentences focusing on COVID-19, SARS-CoV-2, treatments, or pandemic response.

Passage:"""
    
    payload = {
        "model": "tinyllama:latest",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 100
        }
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        generation_time = time.time() - start_time
        
        print(f"✅ Direct API call completed in {generation_time:.2f}s")
        print(f"📝 Response: {result.get('response', 'No response')[:200]}...")
        print(f"📊 Tokens: {result.get('eval_count', 'Unknown')}")
        
        return True, generation_time
        
    except Exception as e:
        print(f"❌ Direct API call failed: {e}")
        return False, 0

def test_llama_index():
    """Test con LlamaIndex Ollama"""
    print("\n🔄 Testing LlamaIndex Ollama...")
    
    llm = Ollama(
        model="tinyllama:latest", 
        thinking=False, 
        request_timeout=30.0,
        temperature=0.1
    )
    
    query = "COVID-19 treatment effectiveness"
    prompt = f"""Write a short scientific passage about COVID-19 research that answers: "{query}"

Write 2-3 sentences focusing on COVID-19, SARS-CoV-2, treatments, or pandemic response.

Passage:"""
    
    start_time = time.time()
    
    try:
        response = llm.complete(prompt, max_tokens=100)
        generation_time = time.time() - start_time
        
        result_text = response.text if hasattr(response, 'text') else str(response)
        
        print(f"✅ LlamaIndex call completed in {generation_time:.2f}s")
        print(f"📝 Response: {result_text[:200]}...")
        
        return True, generation_time
        
    except Exception as e:
        print(f"❌ LlamaIndex call failed: {e}")
        return False, 0

def test_ollama_status():
    """Verifica status Ollama"""
    print("🔄 Checking Ollama status...")
    
    try:
        # Verifica se Ollama è attivo
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            models = response.json()
            print("✅ Ollama is running")
            print(f"📋 Available models: {len(models.get('models', []))}")
            
            # Verifica se il modello specifico è disponibile
            model_names = [m['name'] for m in models.get('models', [])]
            if "tinyllama:latest" in model_names:
                print("✅ tinyllama:latest model found")
            else:
                print("⚠️  tinyllama:latest model not found")
                print(f"Available models: {model_names}")
                return False  # Interrompi se il modello non c'è
            
            return True
        else:
            print(f"❌ Ollama responded with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        return False

def benchmark_multiple_calls():
    """Benchmark multiple chiamate"""
    print("\n🚀 Running benchmark with multiple calls...")
    
    llm = Ollama(
        model="tinyllama:latest", 
        thinking=False, 
        request_timeout=30.0,
        temperature=0.1
    )
    
    queries = [
        "COVID-19 vaccine effectiveness",
        "SARS-CoV-2 transmission mechanisms",
        "COVID-19 treatment protocols",
        "Pandemic response strategies",
        "COVID-19 symptom management"
    ]
    
    times = []
    
    for i, query in enumerate(queries, 1):
        print(f"📝 Query {i}/5: {query}")
        
        prompt = f"Write 2 sentences about: {query}"
        
        start_time = time.time()
        try:
            response = llm.complete(prompt, max_tokens=50)
            generation_time = time.time() - start_time
            times.append(generation_time)
            
            print(f"   ⏱️  {generation_time:.2f}s")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            times.append(999)  # Penalty per errore
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"\n📊 Average generation time: {avg_time:.2f}s")
        print(f"📊 Min: {min(times):.2f}s, Max: {max(times):.2f}s")

def main():
    """Main test function"""
    print("🧪 Ollama Performance Test Script")
    print("=" * 50)
    
    # Test 1: Verifica status
    if not test_ollama_status():
        print("\n❌ Ollama not available. Please start Ollama first.")
        return
    
    # Test 2: API diretta
    direct_success, direct_time = test_ollama_direct()
    
    # Test 3: LlamaIndex
    llamaindex_success, llamaindex_time = test_llama_index()
    
    # Confronto
    if direct_success and llamaindex_success:
        print(f"\n📊 COMPARISON:")
        print(f"Direct API: {direct_time:.2f}s")
        print(f"LlamaIndex: {llamaindex_time:.2f}s")
        print(f"Difference: {abs(llamaindex_time - direct_time):.2f}s")
        
        if llamaindex_time > direct_time * 2:
            print("⚠️  LlamaIndex is significantly slower!")
        elif llamaindex_time > direct_time * 1.5:
            print("⚠️  LlamaIndex has some overhead")
        else:
            print("✅ Performance is similar")
    
    # Test 4: Benchmark multiple calls
    if llamaindex_success:
        benchmark_multiple_calls()
    
    print("\n🏁 Test completed!")

if __name__ == "__main__":
    main()
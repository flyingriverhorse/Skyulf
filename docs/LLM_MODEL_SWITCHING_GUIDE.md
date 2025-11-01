# LLM Model Switching Guide

## Overview
Your MLOps application now supports intelligent model switching for different tasks, with special support for DeepSeek's specialized models.

## Configuration

### 1. Environment Variables
Set up your API keys and model preferences:

```bash
# DeepSeek Configuration
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_API_URL=https://api.deepseek.com
DEEPSEEK_DEFAULT_MODEL=deepseek-chat        # For general chat
DEEPSEEK_CODE_MODEL=deepseek-coder          # For code-related tasks
DEEPSEEK_MATH_MODEL=deepseek-math           # For mathematical analysis

# OpenAI Configuration  
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_DEFAULT_MODEL=gpt-3.5-turbo

# Claude Configuration
ANTHROPIC_API_KEY=your_claude_api_key_here
CLAUDE_DEFAULT_MODEL=claude-3-haiku-20240307
```

### 2. Available Models by Provider

#### DeepSeek Models
- `deepseek-chat` - General conversation and analysis
- `deepseek-coder` - Optimized for code generation and debugging  
- `deepseek-coder-6.7b-base` - Base code model
- `deepseek-coder-6.7b-instruct` - Instruction-tuned code model
- `deepseek-coder-1.3b-base` - Smaller base code model
- `deepseek-coder-1.3b-instruct` - Smaller instruction-tuned code model
- `deepseek-math` - Mathematical problem solving
- `deepseek-reasoner` - Enhanced reasoning capabilities

## Usage Methods

### 1. Browser Console Commands

Open your browser's developer console and use these commands:

```javascript
// Switch to DeepSeek Code model for coding tasks
await window.LLMChat.switchToCodeModel();

// Switch back to regular chat model
await window.LLMChat.switchToChatModel();

// Get model recommendation for specific task
await window.LLMChat.getModelRecommendation('code', 'deepseek');
await window.LLMChat.getModelRecommendation('math', 'deepseek');
await window.LLMChat.getModelRecommendation('analysis');
```

### 2. API Endpoints

#### Get Model Recommendation
```http
POST /llm/recommend-model
Content-Type: application/json

{
    "task_type": "code",
    "provider": "deepseek"
}
```

**Task Types:**
- `chat` - General conversation
- `code` - Programming and development
- `math` - Mathematical analysis
- `analysis` - Data analysis and insights

#### Query with Specific Model
```http
POST /llm/query
Content-Type: application/json

{
    "messages": [
        {"role": "user", "content": "Write a Python function to calculate Fibonacci numbers"}
    ],
    "provider": "deepseek",
    "model": "deepseek-coder",
    "source_id": "your_dataset_id",
    "include_context": true
}
```

## Practical Examples

### Example 1: Code Generation
```javascript
// Switch to code mode
await window.LLMChat.switchToCodeModel();

// Now ask coding questions in the chat:
// "Write a Python function to clean missing values in a pandas DataFrame"
// "Create a scikit-learn pipeline for data preprocessing"
// "Debug this SQL query for performance issues"
```

### Example 2: Mathematical Analysis  
```javascript
// Get math-optimized model
const mathModel = await window.LLMChat.getModelRecommendation('math', 'deepseek');
console.log(`Using ${mathModel.provider} ${mathModel.model} for math tasks`);

// Ask mathematical questions:
// "Explain the statistical significance of these correlation coefficients"
// "What's the optimal sample size for this A/B test?"
```

### Example 3: Data Analysis Tasks
```javascript
// Switch to analysis-optimized model
await window.LLMChat.getModelRecommendation('analysis');

// Ask data analysis questions:
// "Analyze the distribution patterns in this dataset"
// "What feature engineering strategies would improve model performance?"
```

## Model Selection Logic

The system automatically selects optimal models based on task type:

| Task Type | OpenAI | DeepSeek | Claude | Local |
|-----------|---------|----------|---------|-------|
| **chat** | gpt-3.5-turbo | deepseek-chat | claude-3-haiku | llama2 |
| **code** | gpt-4 | deepseek-coder | claude-3-5-sonnet | codellama |
| **math** | gpt-4 | deepseek-math | claude-3-opus | llama2 |
| **analysis** | gpt-4-turbo | deepseek-chat | claude-3-5-sonnet | llama2 |

## Benefits of Using DeepSeek Code Model

### For Code-Related Tasks:
- **Specialized Training**: Optimized specifically for programming tasks
- **Better Code Quality**: More accurate and efficient code generation
- **Language Coverage**: Supports 100+ programming languages
- **Cost Effective**: Often more affordable than GPT-4 for coding tasks

### Use DeepSeek Code When:
- Writing or debugging code
- Explaining programming concepts
- Code reviews and optimization
- API documentation and examples
- Database query optimization
- Data processing scripts

### Use Regular Chat Models When:
- General conversation about data insights
- Business strategy discussions  
- High-level analysis and recommendations
- Executive summaries and reports

## Troubleshooting

### Model Not Switching
1. Check browser console for errors
2. Verify API keys are configured
3. Ensure the provider is available: `await window.LLMChat.getModelRecommendation('code')`

### API Errors
1. Verify API keys in environment variables
2. Check network connectivity
3. Review logs: `/logs/fastapi_app.log`

### Performance Issues
1. DeepSeek models are generally faster for code tasks
2. Use smaller models (1.3B vs 6.7B) for simpler tasks
3. Set appropriate `max_tokens` limits

## Best Practices

1. **Task-Specific Switching**: Use code models for coding, chat models for analysis
2. **Context Management**: Include relevant data context in your queries  
3. **Model Caching**: Model recommendations are cached for performance
4. **Error Handling**: Always check for API errors and fallback models
5. **Cost Optimization**: Use smaller models when sufficient for the task

## Configuration in Code

To programmatically set model preferences in your Python code:

```python
from config import get_settings

settings = get_settings()

# Override default models
settings.DEEPSEEK_DEFAULT_MODEL = "deepseek-coder"  # Use coder as default
settings.DEFAULT_LLM_PROVIDER = "deepseek"          # Make DeepSeek primary

# Get LLM config for service
llm_config = settings.get_llm_config()
```

This setup gives you intelligent model routing that automatically optimizes for your specific use case while maintaining the flexibility to override when needed.
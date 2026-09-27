// server.js - OpenAI to NVIDIA NIM API Proxy (Optimizado para Janitor AI)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware de seguridad para evitar bloqueos de red
app.use(cors());
app.use(express.json());

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://nvidia.com';
const NIM_API_KEY = process.env.NIM_API_KEY;

// Model mapping para conversión automática
const MODEL_MAPPING = {
    'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
    'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
    'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
    'gpt-4o': 'deepseek-ai/deepseek-v3.1',
    'claude-3-opus': 'openai/gpt-oss-120b',
    'claude-3-sonnet': 'openai/gpt-oss-20b',
    'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking',
    
    // Modelos gratuitos nativos de NVIDIA
    'glm-5-3': 'z-ai/glm-5-3',
    'glm-5-3-flash': 'z-ai/glm-5-3-flash',
    'deepseek-flash': 'deepseek-ai/deepseek-v4.1-flash',
    'kumo': 'nvidia/kumo-relational'
};

// Función para procesar la petición del chat
async function handleChatCompletion(req, res) {
  try {
    const { model, messages, temperature, max_tokens } = req.body;
    
    let nimModel = MODEL_MAPPING[model] || model;
    
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 4096,
      stream: false // Forzamos false internamente para evitar que se rompa el flujo de NVIDIA
    };
    
    // Petición directa a la API de NVIDIA
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      }
    });
    
    // Estructura de respuesta exacta compatible con OpenAI y Janitor AI
    const openaiResponse = {
      id: `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model: model,
      choices: response.data.choices.map(choice => ({
        index: choice.index,
        message: { 
          role: choice.message.role, 
          content: choice.message.content || '' 
        },
        finish_reason: choice.finish_reason || 'stop'
      })),
      usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
    };

    // Si Janitor AI pidió streaming, simulamos un paquete de streaming compatible para que no falle
    if (req.body.stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      const chunk = {
        id: openaiResponse.id,
        object: 'chat.completion.chunk',
        created: openaiResponse.created,
        model: openaiResponse.model,
        choices: [{
          index: 0,
          delta: { content: openaiResponse.choices[0].message.content },
          finish_reason: 'stop'
        }]
      };
      
      res.write(`data: ${JSON.stringify(chunk)}\n\n`);
      res.write('data: [DONE]\n\n');
      return res.end();
    }
    
    // Si no tiene streaming activo, responde normal
    res.json(openaiResponse);
    
  } catch (error) {
    console.error('Error en el proxy:', error.message);
    res.status(error.response?.status || 500).json({
      error: { message: error.message || 'Error interno del servidor proxy', type: 'invalid_request_error', code: error.response?.status || 500 }
    });
  }
}

// Configuración de rutas compatibles
app.post('/v1/chat/completions', handleChatCompletion);
app.post('/chat/completions', handleChatCompletion);

app.all('/v1', (req, res) => {
  res.json({ status: 'ok', message: 'NVIDIA Proxy activo para Janitor AI' });
});

app.use((req, res, next) => {
  if (req.path === '/health' || req.path === '/v1/health') {
    return res.json({ status: 'ok', service: 'Proxy adaptado' });
  }
  next();
});

app.all('*', (req, res) => {
  res.status(404).json({
    error: { message: `Ruta ${req.path} no encontrada`, type: 'invalid_request_error', code: 404 }
  });
});

app.listen(PORT, () => {
  console.log(`Proxy corriendo en puerto ${PORT}`);
});

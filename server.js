// server.js - OpenAI to NVIDIA NIM API Proxy (Optimizado para Janitor AI)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// ---------------------------------------------------------------------------
// Configuración
// ---------------------------------------------------------------------------

app.use(cors());
app.use(express.json({ limit: '10mb' }));

// OJO: la URL base real de la API es integrate.api.nvidia.com/v1, NO nvidia.com
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
// Aceptamos cualquiera de los dos nombres de variable, por si en Render
// quedó configurada como NVIDIA_API_KEY en vez de NIM_API_KEY.
const NIM_API_KEY = process.env.NIM_API_KEY || process.env.NVIDIA_API_KEY;

// Validación al arrancar: si falta la API key, avisamos fuerte en los logs
// en vez de fallar en silencio en cada request.
if (!NIM_API_KEY) {
  console.error('[FATAL] Falta la variable de entorno NIM_API_KEY. Configúrala en Render (Environment).');
}

// Cliente axios reutilizable con timeout, para no colgarse si NVIDIA tarda
const nimClient = axios.create({
  baseURL: NIM_API_BASE,
  headers: {
    'Authorization': `Bearer ${NIM_API_KEY}`,
    'Content-Type': 'application/json'
  },
  timeout: 60_000 // 60s
});

// Model mapping para conversión automática.
// IMPORTANTE: estos slugs deben existir tal cual en build.nvidia.com.
// Si un modelo no está en el mapa, se manda tal cual llega (permite usar
// cualquier modelo de NVIDIA directamente sin tener que mapearlo aquí).
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking',

  // Modelos GLM (Z.ai) - confirmados en el catálogo de build.nvidia.com
  'glm-5-3': 'z-ai/glm-5-3',
  'glm-5.1': 'z-ai/glm5.1',
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Da forma de error de OpenAI a cualquier fallo, y deja rastro útil en logs
// (sin filtrar la API key) para poder depurar desde Render → Logs.
function sendError(res, error, context) {
  const status = error.response?.status || 500;
  const nimMessage =
    error.response?.data?.error?.message ||
    error.response?.data?.message ||
    error.message ||
    'Error desconocido en el proxy';

  console.error(`[${new Date().toISOString()}] Error en ${context}:`, {
    status,
    message: nimMessage,
    data: error.response?.data
  });

  res.status(status).json({
    error: {
      message: nimMessage,
      type: 'invalid_request_error',
      code: status
    }
  });
}

// ---------------------------------------------------------------------------
// Chat completions
// ---------------------------------------------------------------------------

async function handleChatCompletion(req, res) {
  const { model, messages, temperature, max_tokens, stream } = req.body || {};

  if (!Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({
      error: { message: '"messages" es requerido y debe ser un array no vacío', type: 'invalid_request_error', code: 400 }
    });
  }

  const nimModel = MODEL_MAPPING[model] || model;
  console.log(`[${new Date().toISOString()}] Modelo pedido por el cliente: "${model}" -> enviando a NVIDIA como: "${nimModel}"`);

  const nimRequest = {
    model: nimModel,
    messages,
    temperature: temperature ?? 0.6,
    max_tokens: max_tokens ?? 4096,
    stream: Boolean(stream)
  };

  try {
    if (stream) {
      // --- Streaming real: reenviamos el stream de NVIDIA tal cual llega ---
      const nimResponse = await nimClient.post('/chat/completions', nimRequest, {
        headers: { 'Accept': 'text/event-stream' },
        responseType: 'stream'
      });

      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      nimResponse.data.pipe(res);

      nimResponse.data.on('error', (err) => {
        console.error('Error durante el streaming:', err.message);
        res.end();
      });

      req.on('close', () => {
        // Si el cliente (Janitor) corta la conexión, cortamos también hacia NVIDIA
        nimResponse.data.destroy();
      });

      return;
    }

    // --- Respuesta normal (no streaming) ---
    const response = await nimClient.post('/chat/completions', nimRequest);

    const openaiResponse = {
      id: `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model: model,
      choices: (response.data.choices || []).map((choice) => ({
        index: choice.index,
        message: {
          role: choice.message?.role || 'assistant',
          content: choice.message?.content || ''
        },
        finish_reason: choice.finish_reason || 'stop'
      })),
      usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
    };

    res.json(openaiResponse);
  } catch (error) {
    sendError(res, error, 'handleChatCompletion');
  }
}

// ---------------------------------------------------------------------------
// Rutas
// ---------------------------------------------------------------------------

app.post('/v1/chat/completions', handleChatCompletion);
app.post('/chat/completions', handleChatCompletion);
// Algunos clientes (como Janitor AI en ciertas configuraciones) postean
// directo a la URL base que les diste, sin agregar /chat/completions.
app.post('/v1', handleChatCompletion);
app.post('/', handleChatCompletion);

// Algunos clientes (y a veces Janitor) consultan /v1/models antes de chatear.
// Devolvemos al menos las claves de nuestro mapping para que esa llamada no falle.
app.get(['/v1/models', '/models'], (req, res) => {
  const ids = Object.keys(MODEL_MAPPING);
  res.json({
    object: 'list',
    data: ids.map((id) => ({ id, object: 'model', owned_by: 'nvidia-nim-proxy' }))
  });
});

app.get(['/health', '/v1/health'], (req, res) => {
  res.json({ status: 'ok', service: 'Proxy NVIDIA NIM adaptado', nimApiBase: NIM_API_BASE });
});

app.get(['/', '/v1'], (req, res) => {
  res.json({ status: 'ok', message: 'Proxy NVIDIA NIM activo para Janitor AI' });
});

app.use((req, res) => {
  res.status(404).json({
    error: { message: `Ruta ${req.method} ${req.path} no encontrada`, type: 'invalid_request_error', code: 404 }
  });
});

app.listen(PORT, () => {
  console.log(`Proxy corriendo en puerto ${PORT}`);
  console.log(`NIM_API_BASE = ${NIM_API_BASE}`);
  console.log(`NIM_API_KEY configurada: ${NIM_API_KEY ? 'sí' : 'NO (falta configurarla)'}`);
});

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
  timeout: 180_000 // 180s: modelos grandes (GLM-5.3) + cold start del free tier de Render pueden tardar
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

  // Modelos GLM (Z.ai) - slugs de API confirmados en la documentación oficial de NVIDIA NIM
  // (ojo: la página web usa guiones en la URL, pero el "model" real de la API lleva punto)
  'glm-5.3': 'z-ai/glm-5.3',
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

  // El cliente puede mandar su propio chat_template_kwargs si quiere
  // forzar el "thinking" on/off; si no manda nada, dejamos el comportamiento
  // por defecto de cada modelo (GLM grande razona, que es lo que da mejores
  // respuestas para roleplay complejo). El silencio mientras razona se
  // resuelve con el heartbeat del streaming, no apagando el razonamiento.
  if (req.body.chat_template_kwargs) {
    nimRequest.chat_template_kwargs = req.body.chat_template_kwargs;
  }

  try {
    if (stream) {
      // Mandamos los headers de streaming YA, antes de esperar a NVIDIA,
      // para que el navegador sepa que la conexión está viva desde el
      // primer instante (importante en modelos con "thinking" largo).
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.flushHeaders?.();

      // Heartbeat: mientras NVIDIA "piensa" y todavía no manda nada,
      // le mandamos un comentario SSE vacío cada 15s. Los comentarios SSE
      // (líneas que arrancan con ":") son ignorados por cualquier parser
      // de eventos, pero mantienen la conexión activa de cara al navegador.
      const heartbeat = setInterval(() => {
        try { res.write(': keep-alive\n\n'); } catch (_) { /* conexión ya cerrada */ }
      }, 15_000);

      // Si el propio cliente (Janitor) cierra la conexión (cancela, regenera,
      // etc.), lo marcamos para no reportar el corte como un error real ni
      // reintentar innecesariamente.
      let clientClosedFirst = false;
      req.on('close', () => { clientClosedFirst = true; });

      async function attemptStream(retriesLeft) {
        let nimResponse;
        try {
          nimResponse = await nimClient.post('/chat/completions', nimRequest, {
            headers: { 'Accept': 'text/event-stream' },
            responseType: 'stream'
          });
        } catch (err) {
          if (retriesLeft > 0 && !clientClosedFirst) {
            console.log(`[${new Date().toISOString()}] Falló al iniciar el streaming, reintentando... (${err.message})`);
            return attemptStream(retriesLeft - 1);
          }
          clearInterval(heartbeat);
          console.error(`[${new Date().toISOString()}] Error iniciando streaming:`, err.message);
          res.write(`data: ${JSON.stringify({ error: { message: err.message, type: 'invalid_request_error' } })}\n\n`);
          return res.end();
        }

        let bytesReceived = 0;
        nimResponse.data.on('data', (chunk) => {
          bytesReceived += chunk.length;
          clearInterval(heartbeat); // ya llegó contenido real, no hace falta más heartbeat
        });

        req.on('close', () => { nimResponse.data.destroy(); });

        nimResponse.data.on('error', (err) => {
          if (clientClosedFirst) {
            console.log(`[${new Date().toISOString()}] Streaming cortado por el cliente (cancelado/regenerado), no es un error.`);
            return res.end();
          }
          if (bytesReceived === 0 && retriesLeft > 0) {
            // NVIDIA cortó antes de mandar nada: reintentamos sin que el cliente se entere.
            console.log(`[${new Date().toISOString()}] NVIDIA cortó la conexión sin enviar datos, reintentando... (${err.message})`);
            return attemptStream(retriesLeft - 1);
          }
          console.error(`[${new Date().toISOString()}] Error inesperado durante el streaming (posible corte del lado de NVIDIA):`, err.message, `| bytes recibidos antes del corte: ${bytesReceived}`);
          res.end();
        });

        nimResponse.data.pipe(res);
      }

      await attemptStream(1); // hasta 1 reintento automático si falla sin haber mandado contenido
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
      

import { SpeechClient } from '@google-cloud/speech';
import { GoogleAuth } from 'google-auth-library';

export const config = {
  api: {
    bodyParser: false,
  },
};

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).send('Method Not Allowed');
  }

  try {
    const auth = new GoogleAuth({
      credentials: {
        client_email: process.env.GOOGLE_CLIENT_EMAIL,
        private_key: process.env.GOOGLE_PRIVATE_KEY.replace(/\\n/g, '\n'),
      },
      scopes: ['https://www.googleapis.com/auth/cloud-platform'],
    });

    const speechClient = new SpeechClient({ auth });

    const audioBytes = await new Promise((resolve, reject) => {
      const chunks = [];
      req.on('data', (chunk) => chunks.push(chunk));
      req.on('end', () => resolve(Buffer.concat(chunks)));
      req.on('error', reject);
    });

    const audio = {
      content: audioBytes.toString('base64'),
    };

    const speechConfig = {
      config: {
        encoding: 'WEBM_OPUS', // MediaRecorderから送られてくる形式
        sampleRateHertz: 48000, // 一般的なマイクのレート
        languageCode: 'ja-JP',
        enableAutomaticPunctuation: true,
      },
      audio: audio,
    };
    
    // ファイルの場合は別のエンコーディングを試す必要があるかもしれません
    // 今回はリアルタイム録音からの復旧を優先します。

    const [response] = await speechClient.recognize(speechConfig);
    const transcription = response.results
      .map((result) => result.alternatives[0].transcript)
      .join('\n');
      
    res.status(200).json({ transcription });

  } catch (error) {
    console.error('Error in transcription API:', error);
    res.status(500).json({ error: 'Transcription failed.' });
  }
}
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import basicSsl from '@vitejs/plugin-basic-ssl'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    basicSsl() // SSL�v���O�C����ǉ�
  ],
  server: {
    host: true, // �l�b�g���[�N�Ɍ��J����
    https: true, // HTTPS��L���ɂ���
    // �ȉ��̃w�b�_�[�ݒ�́A�O��FFmpeg�ŋ�킵���ۂ̖��c�ł����A
    // �Z�L�����e�B�I�ɐ��������ݒ�Ȃ̂Ŏc���Ă����܂��B
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
})
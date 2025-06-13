import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import basicSsl from '@vitejs/plugin-basic-ssl'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    basicSsl() // SSLプラグインを追加
  ],
  server: {
    host: true, // ネットワークに公開する
    https: true, // HTTPSを有効にする
    // 以下のヘッダー設定は、前回FFmpegで苦戦した際の名残ですが、
    // セキュリティ的に推奨される設定なので残しておきます。
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
})
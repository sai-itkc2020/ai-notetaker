import React, { useState, useEffect, useRef } from 'react';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';
import 'react-tabs/style/react-tabs.css';
import ReactMarkdown from 'react-markdown';

// Gemini APIキーは、要約や清書など、ブラウザ側で完結するAI機能のために残しておきます。
const API_KEY = import.meta.env.VITE_GEMINI_API_KEY;
const genAI = new GoogleGenerativeAI(API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash-preview-05-20" });

// クラッシュ復旧のためのIndexedDB操作ヘルパー (変更なし)
const dbManager = {
    dbName: 'TranscriptionDB',
    storeName: 'audioChunks',
    db: null as IDBDatabase | null,

    openDB(): Promise<IDBDatabase> {
        return new Promise((resolve, reject) => {
            if (this.db) {
                resolve(this.db);
                return;
            }
            const request = indexedDB.open(this.dbName, 1);
            request.onerror = () => reject("IndexedDBのオープンに失敗しました");
            request.onsuccess = () => {
                this.db = request.result;
                resolve(this.db);
            };
            request.onupgradeneeded = (event) => {
                const db = (event.target as IDBOpenDBRequest).result;
                if (!db.objectStoreNames.contains(this.storeName)) {
                    db.createObjectStore(this.storeName, { autoIncrement: true });
                }
            };
        });
    },

    async addAudioChunk(chunk: Blob) {
        const db = await this.openDB();
        const transaction = db.transaction(this.storeName, 'readwrite');
        const store = transaction.objectStore(this.storeName);
        store.add(chunk);
        return new Promise<void>((resolve, reject) => {
            transaction.oncomplete = () => resolve();
            transaction.onerror = (event) => reject(`チャンクの保存に失敗しました: ${event.target?.['error']}`);
        });
    },

    async getAllAudioChunks(): Promise<Blob[]> {
        const db = await this.openDB();
        const transaction = db.transaction(this.storeName, 'readonly');
        const store = transaction.objectStore(this.storeName);
        const request = store.getAll();
        return new Promise((resolve, reject) => {
            request.onsuccess = () => resolve(request.result);
            request.onerror = (event) => reject(`チャンクの取得に失敗しました: ${event.target?.['error']}`);
        });
    },

    async clearAudioChunks() {
        const db = await this.openDB();
        const transaction = db.transaction(this.storeName, 'readwrite');
        const store = transaction.objectStore(this.storeName);
        store.clear();
        return new Promise<void>((resolve, reject) => {
            transaction.oncomplete = () => resolve();
            transaction.onerror = (event) => reject(`チャンクのクリアに失敗しました: ${event.target?.['error']}`);
        });
    }
};

// タイムスタンプのフォーマット関数
const formatTime = (seconds: number) => {
    const floorSeconds = Math.floor(seconds);
    const min = Math.floor(floorSeconds / 60);
    const sec = floorSeconds % 60;
    return `[${min.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}]`;
};

// AIによる清書用の関数 (これはブラウザ側でテキスト処理するので残す)
const refineTranscriptWithMemo = async (rawTranscript: string, memo: string) => {
    try {
        const prompt = `あなたは非常に優秀なAI編集者です。
以下の【コンテキスト情報】と【元の文字起こしテキスト】を元に、テキストを清書してください。

【清書ルール】
- 発言者ごとに改行してください。
- 各発言の前に、話している人物名を[名前]の形式で付けてください。名前が不明な場合は[不明]と記載してください。
- 誤字脱字や、明らかな音声認識の間違いがあれば修正してください。
- 全体の意味を変えない範囲で、読みやすいように句読点を調整してください。

【コンテキスト情報】
${memo ? memo : 'なし'}

【元の文字起こしテキスト】
${rawTranscript}

以上の指示に従って、清書されたテキストのみを出力してください。`;

        const result = await model.generateContent(prompt);
        return result.response.text();
    } catch (error) {
        console.error("AIによる清書中にエラー:", error);
        throw error;
    }
}


const App: React.FC = () => {
    const [isRecording, setIsRecording] = useState(false);
    const [transcript, setTranscript] = useState<{ time: number, text: string }[]>([]);
    const [activeMicrophone, setActiveMicrophone] = useState<string>('（マイク未確認）');
    const [downloadLink, setDownloadLink] = useState<string>('');
    const [summary, setSummary] = useState<string>('');
    const [isLoadingAI, setIsLoadingAI] = useState<boolean>(false); // 全てのAI処理中のローディングをこれで管理
    const [copySuccess, setCopySuccess] = useState<string>('');
    const [memoText, setMemoText] = useState<string>('');
    const [activeTab, setActiveTab] = useState(0);
    const [showStopConfirm, setShowStopConfirm] = useState<boolean>(false);
    const [isDarkMode, setIsDarkMode] = useState(false);
    const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
    const [selectedMicId, setSelectedMicId] = useState<string>('');
    const [loadingMessage, setLoadingMessage] = useState('高精度AIの準備をしています...');
    const [modalInfo, setModalInfo] = useState<{ show: boolean, message: string }>({ show: false, message: '' });
    const [recoveryInfo, setRecoveryInfo] = useState<{ show: boolean, chunkCount: number }>({ show: false, chunkCount: 0 });

    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);
    const audioPlayerRef = useRef<HTMLAudioElement | null>(null);
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const animationFrameIdRef = useRef<number | null>(null);
    const fileInputRef = useRef<HTMLInputElement | null>(null);
    const recordingStartTimeRef = useRef<number>(0);
    const memoTextRef = useRef('');
    
    const audioContextRef = useRef<AudioContext | null>(null);
    const analyserRef = useRef<AnalyserNode | null>(null);
    const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);

    useEffect(() => {
        const checkForCrashedData = async () => {
            try {
                const recoveredChunks = await dbManager.getAllAudioChunks();
                if (recoveredChunks.length > 0) {
                    setRecoveryInfo({ show: true, chunkCount: recoveredChunks.length });
                }
            } catch (error) {
                console.error("復旧データの確認中にエラー:", error);
            }
        };

        getAudioDevices();
        checkForCrashedData();
        setLoadingMessage('AI準備完了');
    }, []);
    
    const getAudioDevices = async () => {
        try {
            await navigator.mediaDevices.getUserMedia({ audio: true });
            const audioDevices = (await navigator.mediaDevices.enumerateDevices()).filter(
                (device) => device.kind === 'audioinput'
            );
            setDevices(audioDevices);
            if (audioDevices.length > 0 && !selectedMicId) {
                setSelectedMicId(audioDevices[0].deviceId);
            }
        } catch (err) {
            console.error('マイクデバイスの取得に失敗しました:', err);
        }
    };

    const handleDownload = (content: string, filename: string) => {
        if (!content) return;
        const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    const handleCopyToClipboard = () => {
        if (!summary) return;
        navigator.clipboard.writeText(summary).then(() => {
            setCopySuccess('コピーしました！');
            setTimeout(() => setCopySuccess(''), 2000);
        });
    };

    const stopVisualizer = () => {
        if(animationFrameIdRef.current) {
            cancelAnimationFrame(animationFrameIdRef.current);
        }
        if(audioContextRef.current && audioContextRef.current.state !== 'closed') {
            audioContextRef.current.close();
        }
        if (canvasRef.current) {
            const canvasCtx = canvasRef.current.getContext('2d');
            canvasCtx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        }
    };
    
    const setupVisualizer = (stream: MediaStream) => {
        const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
        const analyser = audioContext.createAnalyser();
        const source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);
        
        audioContextRef.current = audioContext;
        analyserRef.current = analyser;
        sourceRef.current = source;

        analyser.fftSize = 256;
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        const canvas = canvasRef.current;
        if (!canvas) return;
        const canvasCtx = canvas.getContext('2d');

        const draw = () => {
            animationFrameIdRef.current = requestAnimationFrame(draw);
            if (!analyserRef.current || !canvasCtx) return;
            analyserRef.current.getByteFrequencyData(dataArray);
            canvasCtx.fillStyle = isDarkMode ? '#1e1e1e' : '#ffffff';
            canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
            const barWidth = (canvas.width / bufferLength) * 2.5;
            let barHeight;
            let x = 0;
            for (let i = 0; i < bufferLength; i++) {
                barHeight = dataArray[i];
                const r = 200 + (barHeight / 255) * 55;
                const g = 100;
                const b = 180 + (barHeight / 255) * 75;
                canvasCtx.fillStyle = `rgb(${r}, ${g}, ${b})`;
                canvasCtx.fillRect(x, canvas.height - barHeight / 1.5, barWidth, barHeight / 1.5);
                x += barWidth + 1;
            }
        };
        draw();
    };

    // 音声データをバックエンドAPIに送信して文字起こしする共通関数
    const transcribeAudioWithApi = async (audioBlob: Blob) => {
        setIsLoadingAI(true);
        setLoadingMessage("AIが文字起こし中です...");
        try {
            const response = await fetch('/api/transcribe', {
                method: 'POST',
                body: audioBlob,
                headers: {
                    'Content-Type': audioBlob.type,
                },
            });
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: 'サーバーから不明なエラー応答' }));
                throw new Error(errorData.error || '文字起こしAPIリクエストに失敗しました');
            }
            const result = await response.json();
            const elapsedTime = (Date.now() - recordingStartTimeRef.current) / 1000;
            setTranscript([{ time: elapsedTime, text: result.transcription }]);
            setModalInfo({ show: true, message: "文字起こしが完了しました。" });
        } catch (error) {
            console.error("Transcription failed:", error);
            setModalInfo({ show: true, message: `文字起こし中にエラーが発生しました: ${error instanceof Error ? error.message : '不明なエラー'}` });
            setTranscript(prev => [...prev, { time: 0, text: "[文字起こしエラー]" }]);
        } finally {
            setIsLoadingAI(false);
            setLoadingMessage("AI準備完了");
        }
    };


    const toggleRecording = async () => {
        if (isRecording) {
            setShowStopConfirm(true);
        } else {
            try {
                await dbManager.clearAudioChunks();
                const constraints = { audio: { deviceId: selectedMicId ? { exact: selectedMicId } : undefined } };
                const stream = await navigator.mediaDevices.getUserMedia(constraints);
                setActiveMicrophone(devices.find(d => d.deviceId === selectedMicId)?.label || 'デフォルトマイク');
                setupVisualizer(stream);
                setDownloadLink('');
                audioChunksRef.current = [];
                
                const options = { mimeType: 'audio/webm;codecs=opus' };
                mediaRecorderRef.current = new MediaRecorder(stream, options);
                
                mediaRecorderRef.current.ondataavailable = (e) => {
                    if (e.data.size > 0) {
                        audioChunksRef.current.push(e.data);
                        dbManager.addAudioChunk(e.data); // クラッシュ復旧用に保存
                    }
                };

                mediaRecorderRef.current.onstop = () => {
                    const audioBlob = new Blob(audioChunksRef.current, { type: options.mimeType });
                    setDownloadLink(URL.createObjectURL(audioBlob));
                    stream.getTracks().forEach(track => track.stop());
                    if (audioBlob.size > 0) {
                        transcribeAudioWithApi(audioBlob); // 録音停止時にAPIに送信
                    }
                };
                
                setTranscript([]);
                setSummary('');
                setActiveTab(0);
                setIsRecording(true);
                recordingStartTimeRef.current = Date.now();
                mediaRecorderRef.current.start(10000); // 10秒ごとにデータを収集

            } catch (err) {
                console.error("マイクアクセス失敗:", err);
                setActiveMicrophone('マイクへのアクセスが拒否されました。');
            }
        }
    };
    
    const handleConfirmStop = () => {
        if (mediaRecorderRef.current?.state === "recording") {
            mediaRecorderRef.current.stop();
        }
        setIsRecording(false);
        stopVisualizer();
        setShowStopConfirm(false);
    };

    const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;
        setTranscript([]);
        setSummary('');
        setDownloadLink(URL.createObjectURL(file));
        setActiveTab(0);
        await transcribeAudioWithApi(file); // ファイルをそのままAPIに送信
    };
    
    const handleProcessRecoveredData = async () => {
        setRecoveryInfo({ show: false, chunkCount: 0 });
        const recoveredChunks = await dbManager.getAllAudioChunks();
        if (recoveredChunks.length > 0) {
            const recoveredBlob = new Blob(recoveredChunks, { type: 'audio/webm;codecs=opus' });
            setDownloadLink(URL.createObjectURL(recoveredBlob));
            await transcribeAudioWithApi(recoveredBlob); // 復旧データもAPIに送信
        }
        dbManager.clearAudioChunks();
    };


    const handleRefineTranscript = async () => {
        if(transcript.length === 0) {
            setModalInfo({ show: true, message: '清書する文字起こしデータがありません。'});
            return;
        }
        setIsLoadingAI(true);
        setLoadingMessage('AIが清書中です...');
        try {
            const rawTranscript = transcript.map(t => t.text).join('\n');
            const refinedText = await refineTranscriptWithMemo(rawTranscript, memoText);
            
            const refinedLines = refinedText.split('\n').filter(line => line.trim() !== '');
            const newTranscript = refinedLines.map((line, index) => {
                const originalEntry = transcript[index] || transcript[transcript.length - 1];
                return { time: originalEntry.time, text: line };
            });

            setTranscript(newTranscript);
            setModalInfo({ show: true, message: 'AIによる清書が完了しました。'});
        } catch (error) {
            setModalInfo({ show: true, message: 'AIによる清書中にエラーが発生しました。'});
        } finally {
            setIsLoadingAI(false);
            setLoadingMessage('AI準備完了');
        }
    };
    
    const handleTimestampClick = (time: number) => { 
        if(audioPlayerRef.current) {
            audioPlayerRef.current.currentTime = time;
            audioPlayerRef.current.play();
        }
    };
    
    const generateSummary = async () => {
        const plainTranscript = transcript.map(item => `${formatTime(item.time)} ${item.text}`).join('\n\n');
        if (!plainTranscript && !memoText) {
            setModalInfo({ show: true, message: '要約する文字起こしやメモがありません。'});
            return;
        }
        setIsLoadingAI(true);
        setLoadingMessage('AIが議事録を作成中です...');
        setSummary('');
        setActiveTab(1);
        try {
            // ... (要約プロンプトは変更なし) ...
            const prompt = `あなたはプロの議事録作成AIです。提供された以下のタイムスタンプ付き【会議の文字起こし】と【手動メモ】を分析し、その内容から網羅的かつ簡潔な議事録を作成してください。議事録は以下の構造とルールに従って記述してください。---## 議事録### 1. 会議概要**会議日時**: ${new Date().toLocaleString('ja-JP')}会議の目的、主要な議題、および全体的な結論を簡潔にまとめる。### 2. 議論の要点会議で話し合われた重要なポイントや論点を、主要なテーマごとに整理して箇条書きで記述する。### 3. 決定事項会議で合意された事項や結論を明確に箇条書きで記述する。### 4. アクションアイテム (次回以降のタスク)会議で決定された具体的な行動やタスク、担当者、期限を箇条書きで記述する。---**【会議の文字起こし】**${plainTranscript}**【手動メモ】**${memoText}`;
            const result = await model.generateContent(prompt);
            setSummary(result.response.text());
            setModalInfo({ show: true, message: '議事録が完成しました！「AIによる議事録」タブでご確認ください。'});
        } catch (error) {
            console.error("AIによる要約中にエラーが発生しました:", error);
            setModalInfo({ show: true, message: '議事録の生成中にエラーが発生しました。'});
        } finally {
            setIsLoadingAI(false);
            setLoadingMessage("AI準備完了");
        }
    };
    
    const customStyles = `
        body { background-color: ${isDarkMode ? '#121212' : '#f4f7f9'}; color: ${isDarkMode ? '#e0e0e0' : '#333'}; transition: background-color 0.3s, color 0.3s; }
        .main-container { background-color: ${isDarkMode ? '#1e1e1e' : '#ffffff'}; padding: 20px 30px; font-family: sans-serif; max-width: 800px; margin: 20px auto; border-radius: 8px; box-shadow: ${isDarkMode ? '0 4px 20px rgba(0,0,0,0.5)' : '0 4px 20px rgba(0,0,0,0.1)'}; transition: background-color 0.3s, box-shadow 0.3s; }
        h1, h2 { color: ${isDarkMode ? '#ffffff' : '#000000'}; }
        .info-box { background-color: ${isDarkMode ? '#2a2a2a' : '#f0f0f0'}; border: 1px solid ${isDarkMode ? '#444' : '#ddd'}; color: ${isDarkMode ? '#e0e0e0' : '#333'}; }
        .react-tabs__tab { background: ${isDarkMode ? '#2a2a2a' : '#f0f0f0'}; border-color: ${isDarkMode ? '#444' : '#ddd'}; color: ${isDarkMode ? '#a0a0a0' : '#333'}; border-bottom: none; }
        .react-tabs__tab--selected { background: ${isDarkMode ? '#1e1e1e' : '#ffffff'}; color: ${isDarkMode ? 'white' : '#007bff'}; border-color: ${isDarkMode ? '#444' : '#ddd'}; border-bottom: 1px solid ${isDarkMode ? '#1e1e1e' : '#ffffff'}; position: relative; top: 1px; }
        .react-tabs__tab-panel--selected { border: 1px solid ${isDarkMode ? '#444' : '#ddd'}; padding: 15px; background-color: ${isDarkMode ? '#2a2a2a' : '#ffffff'}; }
        .transcript-panel { background-color: ${isDarkMode ? '#2a2a2a' : '#fdfdfd'}; border: 1px solid ${isDarkMode ? '#444' : '#ccc'}; white-space: pre-wrap; }
        textarea { background-color: ${isDarkMode ? '#333' : '#fff'}; color: ${isDarkMode ? '#e0e0e0' : '#000'}; border-color: ${isDarkMode ? '#555' : '#ccc'}; }
        .ai-summary-panel { background-color: #1e1e1e; color: #d4d4d4; padding: 25px; border-radius: 8px; border: 1px solid #333; min-height: 250px; white-space: pre-wrap; line-height: 1.7; }
        .ai-summary-panel h2, .ai-summary-panel h3 { color: #ffffff; }
        .ai-summary-panel strong { color: #569cd6; }
        .toggle-switch { display: flex; flex-direction: column; align-items: center; gap: 4px; }
        .switch { position: relative; display: inline-block; width: 50px; height: 26px; }
        .switch input { opacity: 0; width: 0; height: 0; }
        .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #ccc; transition: .4s; border-radius: 34px; }
        .slider:before { position: absolute; content: ""; height: 20px; width: 20px; left: 3px; bottom: 3px; background-color: white; transition: .4s; border-radius: 50%; }
        input:checked + .slider { background-color: #2196F3; }
        input:checked + .slider:before { transform: translateX(24px); }
        select { padding: 8px; border-radius: 5px; border: 1px solid ${isDarkMode ? '#444' : '#ddd'}; background-color: ${isDarkMode ? '#2a2a2a' : '#f0f0f0'}; color: ${isDarkMode ? '#e0e0e0' : '#333'}; }
        .timestamp { color: #ff69b4; cursor: pointer; font-weight: bold; margin-right: 10px; }
        .timestamp:hover { text-decoration: underline; }
        audio { filter: ${isDarkMode ? 'invert(1) contrast(0.8) brightness(1.2)' : 'none'}; }
    `;

    return (
        <>
            <style>{customStyles}</style>
            <div className="main-container">
                {/* ... (JSX部分はほぼ変更なし、ただしボタンのdisabledロジックをisLoadingAIに連動させる) ... */}
                <div style={{display: 'flex', gap: '10px'}}>
                    <button onClick={toggleRecording} disabled={isLoadingAI} style={{ /* ... */ }}>
                        {isRecording ? '■ 録音停止' : (isLoadingAI ? loadingMessage : '● 録音開始')}
                    </button>
                    <input type="file" ref={fileInputRef} onChange={handleFileChange} accept="audio/*" style={{ display: 'none' }} />
                    <button onClick={() => fileInputRef.current?.click()} disabled={isLoadingAI || isRecording} style={{ /* ... */ }}>
                        {isLoadingAI ? loadingMessage : '音声ファイルを読み込む'}
                    </button>
                </div>

                 {/* ... (残りのJSXも同様にisLoadingAIでローディング状態を制御) ... */}
                 
            </div>

            {/* ... (モーダル部分は変更なし) ... */}
        </>
    );
};

export default App;
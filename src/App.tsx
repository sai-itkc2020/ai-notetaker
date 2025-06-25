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
    const [isLoadingAI, setIsLoadingAI] = useState<boolean>(false);
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
        setTranscript([]); // 古い結果をクリア
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
            const elapsedTime = (Date.now() - (recordingStartTimeRef.current || Date.now())) / 1000;
            setTranscript([{ time: elapsedTime, text: result.transcription }]);
            setModalInfo({ show: true, message: "文字起こしが完了しました。" });

        } catch (error) {
            console.error("Transcription failed:", error);
            setModalInfo({ show: true, message: `ファイルの処理中にエラーが発生しました。お使いのブラウザが対応していない音声形式の可能性があります。` });
            setTranscript([]);
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
                        dbManager.addAudioChunk(e.data);
                    }
                };

                mediaRecorderRef.current.onstop = () => {
                    const audioBlob = new Blob(audioChunksRef.current, { type: options.mimeType });
                    setDownloadLink(URL.createObjectURL(audioBlob));
                    stream.getTracks().forEach(track => track.stop());
                    if (audioBlob.size > 0) {
                        transcribeAudioWithApi(audioBlob);
                    }
                };
                
                setTranscript([]);
                setSummary('');
                setActiveTab(0);
                setIsRecording(true);
                recordingStartTimeRef.current = Date.now();
                mediaRecorderRef.current.start(10000);

            } catch (err) {
                console.error("マイクアクセス失敗:", err);
                setModalInfo({ show: true, message: `マイクへのアクセスに失敗しました: ${err instanceof Error ? err.message : '不明なエラー'}` });
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
        await transcribeAudioWithApi(file);
    };
    
    const handleProcessRecoveredData = async () => {
        setRecoveryInfo({ show: false, chunkCount: 0 });
        const recoveredChunks = await dbManager.getAllAudioChunks();
        if (recoveredChunks.length > 0) {
            const recoveredBlob = new Blob(recoveredChunks, { type: 'audio/webm;codecs=opus' });
            setDownloadLink(URL.createObjectURL(recoveredBlob));
            await transcribeAudioWithApi(recoveredBlob);
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
            const meetingDate = new Date().toLocaleString('ja-JP');
            const prompt = `あなたはプロの議事録作成AIです。提供された以下のタイムスタンプ付き【会議の文字起こし】と【手動メモ】を分析し、その内容から網羅的かつ簡潔な議事録を作成してください。議事録は以下の構造とルールに従って記述してください。---## 議事録### 1. 会議概要**会議日時**: ${meetingDate}会議の目的、主要な議題、および全体的な結論を簡潔にまとめる。### 2. 議論の要点会議で話し合われた重要なポイントや論点を、主要なテーマごとに整理して箇条書きで記述する。### 3. 決定事項会議で合意された事項や結論を明確に箇条書きで記述する。### 4. アクションアイテム (次回以降のタスク)会議で決定された具体的な行動やタスク、担当者、期限を箇条書きで記述する。---**【会議の文字起こし】**${plainTranscript}**【手動メモ】**${memoText}`;
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
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="22"></line></svg>
                        <h1 style={{ fontSize: '1.8em', margin: 0 }}>AI議事録ツール</h1>
                    </div>
                    <div className="toggle-switch">
                        <span style={{ fontSize: '12px', color: isDarkMode ? '#aaa' : '#555' }}>ダークモード</span>
                        <label className="switch">
                            <input type="checkbox" checked={isDarkMode} onChange={() => setIsDarkMode(!isDarkMode)} />
                            <span className="slider"></span>
                        </label>
                    </div>
                </div>

                <p style={{marginTop: 0}}>その場で録音するか、既存の音声ファイルを読み込んでください。</p>
                <div style={{display: 'flex', gap: '10px'}}>
                    <button onClick={toggleRecording} disabled={isLoadingAI} style={{ fontSize: '16px', padding: '10px 20px', backgroundColor: isRecording ? '#dc3545' : (isLoadingAI ? '#6c757d' : '#007bff'), color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer', flex: 1 }}>
                        {isRecording ? '■ 録音停止' : (isLoadingAI ? loadingMessage : '● 録音開始')}
                    </button>
                    <input type="file" ref={fileInputRef} onChange={handleFileChange} accept="audio/*,video/mp4" style={{ display: 'none' }} />
                    <button onClick={() => fileInputRef.current?.click()} disabled={isLoadingAI || isRecording} style={{ fontSize: '16px', padding: '10px 20px', backgroundColor: isLoadingAI || isRecording ? '#6c757d' : '#6c757d', color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer', flex: 1 }}>
                        {isLoadingAI ? loadingMessage : '音声ファイルを読み込む'}
                    </button>
                </div>
                
                <canvas ref={canvasRef} style={{ width: '100%', height: '60px', marginTop: '15px', borderRadius: '5px', display: isRecording ? 'block' : 'none' }}></canvas>
                
                <div className="info-box" style={{ marginTop: '15px', padding: '15px', borderRadius: '5px' }}>
                     <label htmlFor="mic-select" style={{display: 'block', marginBottom: '8px'}}><strong>使用するマイクを選択</strong></label>
                     <select id="mic-select" value={selectedMicId} onChange={(e) => setSelectedMicId(e.target.value)} disabled={isRecording || isLoadingAI} style={{width: '100%'}}>
                         {devices.length === 0 && <option>マイクが見つかりません</option>}
                         {devices.map(device => ( <option key={device.deviceId} value={device.deviceId}> {device.label || `マイク ${devices.indexOf(device) + 1}`} </option> ))}
                     </select>
                </div>

                <div style={{ margin: '20px 0', textAlign: 'center' }}>
                    <button onClick={generateSummary} disabled={isLoadingAI || transcript.length === 0} style={{ fontSize: '16px', padding: '12px 24px', backgroundColor: isLoadingAI || transcript.length === 0 ? '#6c757d' : '#17a2b8', color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer', transition: 'background-color 0.3s' }}>
                        {isLoadingAI ? 'AIが考え中...' : '🤖 AIで議事録を作成'}
                    </button>
                </div>

                <Tabs selectedIndex={activeTab} onSelect={index => setActiveTab(index)} style={{marginTop: '20px'}}>
                    <TabList>
                        <Tab>文字起こし</Tab>
                        <Tab>AIによる議事録</Tab>
                        <Tab>手動メモ</Tab>
                    </TabList>

                    <TabPanel>
                        {downloadLink && ( <div style={{ margin: '15px 0' }}> <p style={{fontWeight: 'bold', marginBottom: '5px'}}>音声ファイルの再生</p> <audio ref={audioPlayerRef} src={downloadLink} controls style={{width: '100%'}} /> <a href={downloadLink} download={`recording-${new Date().toISOString().slice(0,10)}.webm`} style={{fontSize: '12px', display: 'block', textAlign: 'right', marginTop: '5px'}}>ダウンロード</a> </div> )}
                        
                        <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', marginBottom: '10px'}}>
                            <h2 style={{ marginTop: '10px', marginBottom: '10px' }}>文字起こし結果</h2>
                            {transcript.length > 0 && !isLoadingAI && (
                                <button 
                                    onClick={handleRefineTranscript}
                                    style={{
                                        fontSize: '14px', 
                                        padding: '8px 16px', 
                                        backgroundColor: '#28a745', 
                                        color: 'white', 
                                        border: 'none', 
                                        borderRadius: '5px', 
                                        cursor: 'pointer' 
                                    }}>
                                    AIで清書する
                                </button>
                            )}
                            <button onClick={() => handleDownload(transcript.map(t => `${formatTime(t.time)} ${t.text}`).join('\n\n'), `transcript-${new Date().toISOString().slice(0, 10)}.txt`)} disabled={transcript.length === 0} style={{fontSize: '14px', padding: '8px 16px', backgroundColor: transcript.length === 0 ? '#6c757d' : '#6f42c1', color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer' }}>
                                原文をテキストでダウンロード
                            </button>
                        </div>
                        <div className="transcript-panel" style={{ padding: '10px', minHeight: '200px', lineHeight: '1.8' }}>
                            {isLoadingAI && <p>{loadingMessage}</p>}
                            {!isLoadingAI && transcript.length > 0 ? (
                                transcript.map((item, index) => (
                                    <p key={index} style={{margin: '0 0 10px 0'}}>
                                        <span className="timestamp" onClick={() => handleTimestampClick(item.time)}> {formatTime(item.time)} </span>
                                        {item.text}
                                    </p>
                                ))
                            ) : (
                                !isLoadingAI && <p>ここに高精度AIによる文字起こし結果がリアルタイムで表示されます...</p>
                            )}
                        </div>
                    </TabPanel>
                    
                    <TabPanel>
                        <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', marginBottom: '15px'}}>
                            <h2 style={{ marginTop: '10px', marginBottom: '10px' }}>AIによる議事録</h2>
                            <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                                <button onClick={() => handleDownload(summary, `minutes-${new Date().toISOString().slice(0, 10)}.txt`)} disabled={!summary} style={{fontSize: '14px', padding: '8px 16px', backgroundColor: !summary ? '#6c757d' : '#0069d9', color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer' }}> 議事録をダウンロード </button>
                                <button onClick={handleCopyToClipboard} disabled={!summary} style={{fontSize: '14px', padding: '8px 16px', backgroundColor: !summary ? '#6c757d' : '#5a6268', color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer' }}> 議事録をコピー </button>
                                {copySuccess && <span style={{color: 'green', fontSize: '14px'}}>{copySuccess}</span>}
                            </div>
                        </div>
                        <div className="ai-summary-panel">
                            {isLoadingAI && summary === '' && <p>AIが議事録を作成中です...</p>}
                            {summary ? <ReactMarkdown>{summary}</ReactMarkdown> : !isLoadingAI && <p>ここにAIが生成した議事録が表示されます...</p>}
                        </div>
                    </TabPanel>

                    <TabPanel>
                        <h2 style={{ marginTop: '10px' }}>手動メモ</h2>
                        <textarea
                            value={memoText}
                            onChange={(e) => {
                                setMemoText(e.target.value);
                                memoTextRef.current = e.target.value;
                            }}
                            placeholder="会議の参加者、決定事項の背景、次のアクションなど、音声以外の情報をここにメモします。例：参加者：山田太郎、佐藤花子"
                            style={{ width: '98%', minHeight: '250px', padding: '10px', border: '1px solid', borderRadius: '5px', fontSize: '16px', lineHeight: '1.6' }}
                        />
                    </TabPanel>
                </Tabs>
            </div>

            {modalInfo.show && (
                <div style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', backgroundColor: 'rgba(0, 0, 0, 0.5)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 1000 }}>
                    <div style={{ backgroundColor: isDarkMode ? '#2a2a2a' : 'white', color: isDarkMode ? '#e0e0e0' : '#333', padding: '25px 30px', borderRadius: '8px', boxShadow: '0 4px 15px rgba(0,0,0,0.2)', textAlign: 'center', width: '90%', maxWidth: '400px' }}>
                        <p style={{margin: '0 0 20px', fontSize: '1.1em', lineHeight: '1.6'}}>{modalInfo.message}</p>
                        <button onClick={() => setModalInfo({ show: false, message: '' })} style={{fontSize: '15px', padding: '10px 25px', backgroundColor: '#007bff', color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer'}} >
                            閉じる
                        </button>
                    </div>
                </div>
            )}
            
            {recoveryInfo.show && (
                <div style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', backgroundColor: 'rgba(0, 0, 0, 0.6)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 1001 }}>
                    <div style={{ backgroundColor: isDarkMode ? '#2a2a2a' : 'white', color: isDarkMode ? '#e0e0e0' : '#333', padding: '25px', borderRadius: '8px', boxShadow: '0 4px 15px rgba(0,0,0,0.2)', textAlign: 'center', width: '90%', maxWidth: '420px' }}>
                        <h3 style={{marginTop: 0, fontSize: '1.3em', color: '#ffc107'}}>⚠ 未保存の録音データ</h3>
                        <p style={{margin: '15px 0 25px', fontSize: '0.95em'}}>前回のセッションが正常に終了されませんでした。途中まで録音されたデータが見つかりましたが、どうしますか？</p>
                        <div style={{display: 'flex', justifyContent: 'center', gap: '15px'}}>
                            <button onClick={handleProcessRecoveredData} style={{fontSize: '15px', padding: '10px 20px', backgroundColor: '#28a745', color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer', minWidth: '120px'}}>
                                文字起こしする
                            </button>
                            <button onClick={() => { dbManager.clearAudioChunks(); setRecoveryInfo({ show: false, chunkCount: 0 }); }} style={{fontSize: '15px', padding: '10px 20px', backgroundColor: '#6c757d', color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer', minWidth: '120px'}}>
                                データを破棄
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {showStopConfirm && (
                <div style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', backgroundColor: 'rgba(0, 0, 0, 0.6)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 1000 }}>
                    <div style={{ backgroundColor: isDarkMode ? '#2a2a2a' : 'white', color: isDarkMode ? '#e0e0e0' : '#333', padding: '25px', borderRadius: '8px', boxShadow: '0 4px 15px rgba(0,0,0,0.2)', textAlign: 'center', width: '90%', maxWidth: '360px' }}>
                        <h3 style={{marginTop: 0, fontSize: '1.3em', color: isDarkMode ? '#ffffff' : '#000000'}}>録音を終了しますか？</h3>
                        <p style={{margin: '15px 0 25px', fontSize: '0.95em'}}>録音を終了し、議事録の作成準備を開始します。</p>
                        <div style={{display: 'flex', justifyContent: 'center', gap: '15px'}}>
                            <button onClick={handleConfirmStop} style={{fontSize: '15px', padding: '10px 20px', backgroundColor: '#dc3545', color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer', minWidth: '110px'}}>
                                はい、終了する
                            </button>
                            <button onClick={() => setShowStopConfirm(false)} style={{fontSize: '15px', padding: '10px 20px', backgroundColor: '#6c757d', color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer', minWidth: '110px'}}>
                                キャンセル
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </>
    );
};

export default App;
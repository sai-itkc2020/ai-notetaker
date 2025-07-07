import React, { useState, useEffect, useRef } from 'react';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';
import 'react-tabs/style/react-tabs.css';
import ReactMarkdown from 'react-markdown';

//【重要】APIキーを安全な環境変数から読み込みます
const API_KEY = import.meta.env.VITE_GEMINI_API_KEY;

let genAI: GoogleGenerativeAI | null = null;
try {
    if (API_KEY) {
        genAI = new GoogleGenerativeAI(API_KEY);
    } else {
        console.error("APIキーが設定されていません。");
    }
} catch (error) {
    console.error("GoogleGenerativeAIの初期化に失敗:", error);
}

const model = genAI ? genAI.getGenerativeModel({ model: "gemini-2.5-flash-preview-05-20" }) : null;

// クラッシュ復旧のためのIndexedDB操作ヘルパー
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
            transaction.onerror = (event) => reject(`チャンクの保存に失敗しました: ${(event.target as any)?.error}`);
        });
    },
    async getAllAudioChunks(): Promise<Blob[]> {
        const db = await this.openDB();
        const transaction = db.transaction(this.storeName, 'readonly');
        const store = transaction.objectStore(this.storeName);
        const request = store.getAll();
        return new Promise((resolve, reject) => {
            request.onsuccess = () => resolve(request.result);
            request.onerror = (event) => reject(`チャンクの取得に失敗しました: ${(event.target as any)?.error}`);
        });
    },
    async clearAudioChunks() {
        const db = await this.openDB();
        const transaction = db.transaction(this.storeName, 'readwrite');
        const store = transaction.objectStore(this.storeName);
        store.clear();
        return new Promise<void>((resolve, reject) => {
            transaction.oncomplete = () => resolve();
            transaction.onerror = (event) => reject(`チャンクのクリアに失敗しました: ${(event.target as any)?.error}`);
        });
    }
};

const blobToBase64 = (blob: Blob): Promise<string> => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => {
            const base64data = reader.result as string;
            resolve(base64data.split(',')[1]);
        };
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
};

// ★★★ 改善点：言語設定を引数で受け取るように変更 ★★★
const transcribeFileRaw = async (audioChunk: Blob, language: 'ja' | 'en' | 'auto') => {
    if (!model) throw new Error("モデルが初期化されていません。");
    try {
        const audioBase64 = await blobToBase64(audioChunk);
        let prompt = '';
        if (language === 'ja') {
            prompt = `以下の音声データを日本語で文字起こししてください。話者特定の必要はありません。句読点のみ適切に付与してください。`;
        } else if (language === 'en') {
            prompt = `Please transcribe the following audio data in English. Speaker identification is not necessary. Just add punctuation appropriately.`;
        } else {
            prompt = `Identify the primary language in the following audio data and transcribe it accurately in that language. Do not identify speakers. Just add punctuation appropriately.`;
        }
        
        const result = await model.generateContent([ prompt, { inlineData: { mimeType: 'audio/wav', data: audioBase64 } } ]);
        const text = result.response.text();
        return (!text || text.trim() === '') ? '（無音または認識不能区間）' : text;
    } catch (error) {
        console.error("ファイルからの初期文字起こし中にエラー:", error);
        throw error;
    }
};

// ★★★ 改善点：言語設定を引数で受け取るように変更 ★★★
const refineTranscriptWithMemo = async (rawTranscript: string, memo: string, language: 'ja' | 'en' | 'auto') => {
    if (!model) throw new Error("モデルが初期化されていません。");
     try {
        const langInstruction = language === 'en' 
            ? 'Please act as an expert editor. Refine the following text based on the context and original transcript provided below. Rules: Start each speaker on a new line. Prefix each utterance with the speaker\'s name in the format [Name]. If the name is unknown, use [Unknown]. Correct typos and obvious speech recognition errors. Adjust punctuation for readability without changing the overall meaning.'
            : 'あなたは非常に優秀なAI編集者です。以下の【コンテキスト情報】と【元の文字起こしテキスト】を元に、テキストを清書してください。【清書ルール】- 発言者ごとに改行してください。- 各発言の前に、話している人物名を[名前]の形式で付けてください。名前が不明な場合は[不明]と記載してください。- 誤字脱字や、明らかな音声認識の間違いがあれば修正してください。- 全体の意味を変えない範囲で、読みやすいように句読点を調整してください。';

        const contextLabel = language === 'en' ? 'Contextual Information' : 'コンテキスト情報';
        const transcriptLabel = language === 'en' ? 'Original Transcript' : '元の文字起こしテキスト';
        const finalInstruction = language === 'en' ? 'Following the instructions above, please output only the refined text.' : '以上の指示に従って、清書されたテキストのみを出力してください。';

        const prompt = `${langInstruction}\n\n【${contextLabel}】\n${memo ? memo : 'なし'}\n\n【${transcriptLabel}】\n${rawTranscript}\n\n${finalInstruction}`;

        const result = await model.generateContent(prompt);
        return result.response.text();
    } catch (error) {
        console.error("AIによる清書中にエラー:", error);
        throw error;
    }
}

const bufferToWav = (buffer: AudioBuffer): Blob => {
    const numOfChan = buffer.numberOfChannels;
    const len = buffer.length * numOfChan * 2 + 44;
    const bufferOut = new ArrayBuffer(len);
    const view = new DataView(bufferOut);
    let pos = 0;
    const writeString = (s: string) => { for (let i = 0; i < s.length; i++) { view.setUint8(pos++, s.charCodeAt(i)); } };
    const setUint16 = (data: number) => { view.setUint16(pos, data, true); pos += 2; };
    const setUint32 = (data: number) => { view.setUint32(pos, data, true); pos += 4; };
    writeString('RIFF'); setUint32(len - 8); writeString('WAVE');
    writeString('fmt '); setUint32(16); setUint16(1);
    setUint16(numOfChan); setUint32(buffer.sampleRate);
    setUint32(buffer.sampleRate * 2 * numOfChan); setUint16(numOfChan * 2);
    setUint16(16); writeString('data'); setUint32(len - pos - 4);
    const channels = [];
    for (let i = 0; i < numOfChan; i++) { channels.push(buffer.getChannelData(i)); }
    let offset = 0;
    while (pos < len) {
        for (let i = 0; i < numOfChan; i++) {
            let sample = Math.max(-1, Math.min(1, channels[i][offset]));
            sample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
            view.setInt16(pos, sample, true);
            pos += 2;
        }
        offset++;
    }
    return new Blob([view], { type: 'audio/wav' });
};

const formatTime = (seconds: number) => {
    const floorSeconds = Math.floor(seconds);
    const min = Math.floor(floorSeconds / 60);
    const sec = floorSeconds % 60;
    return `[${min.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}]`;
};

const App: React.FC = () => {
    const [isRecording, setIsRecording] = useState(false);
    const [transcript, setTranscript] = useState<{ time: number, text: string }[]>([]);
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
    // ★★★ 追加：状態管理の追加 ★★★
    const [language, setLanguage] = useState<'ja' | 'en' | 'auto'>('auto');
    const [isRefining, setIsRefining] = useState(false);
    const [isProcessingFile, setIsProcessingFile] = useState(false);

    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);
    const audioPlayerRef = useRef<HTMLAudioElement | null>(null);
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const animationFrameIdRef = useRef<number | null>(null);
    const fileInputRef = useRef<HTMLInputElement | null>(null);
    const memoTextRef = useRef('');
    const audioContextRef = useRef<AudioContext | null>(null);
    // ★★★ 追加：マイクチェック用とファイル処理中断用のRef ★★★
    const micCheckStreamRef = useRef<MediaStream | null>(null);
    const fileAbortControllerRef = useRef<AbortController | null>(null);

    useEffect(() => {
        const init = async () => {
            await getAudioDevices(true); // 初回のみマイクチェックを開始
            await checkForCrashedData();
            setLoadingMessage('AI準備完了');
        };
        init();
    }, []);

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

    const getAudioDevices = async (startMicCheck = false) => {
        try {
            await navigator.mediaDevices.getUserMedia({ audio: true });
            const audioDevices = (await navigator.mediaDevices.enumerateDevices()).filter(
                (device) => device.kind === 'audioinput'
            );
            setDevices(audioDevices);
            if (audioDevices.length > 0) {
                const defaultMicId = audioDevices[0].deviceId;
                if (!selectedMicId) {
                    setSelectedMicId(defaultMicId);
                }
                if (startMicCheck) {
                    handleMicChange(defaultMicId);
                }
            }
        } catch (err) {
            console.error('マイクデバイスの取得に失敗しました:', err);
            setModalInfo({ show: true, message: 'マイクへのアクセスに失敗しました。ブラウザの設定でマイクの使用を許可してください。' });
        }
    };

    // ★★★ 改善点：マイクの事前確認機能 ★★★
    const handleMicChange = async (deviceId: string) => {
        setSelectedMicId(deviceId);
        stopVisualizer(true); // 既存のビジュアライザーを停止
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: { deviceId: { exact: deviceId } } });
            micCheckStreamRef.current = stream;
            setupVisualizer(stream);
        } catch (err) {
            console.error('選択されたマイクの取得に失敗:', err);
            setModalInfo({ show: true, message: '選択されたマイクの起動に失敗しました。' });
        }
    };
    
    const stopVisualizer = (isMicCheckOnly = false) => {
        if(animationFrameIdRef.current) {
            cancelAnimationFrame(animationFrameIdRef.current);
            animationFrameIdRef.current = null;
        }
        if (micCheckStreamRef.current) {
            micCheckStreamRef.current.getTracks().forEach(track => track.stop());
            micCheckStreamRef.current = null;
        }
        if (!isMicCheckOnly && audioContextRef.current && audioContextRef.current.state !== 'closed') {
            audioContextRef.current.close();
        }
        if (canvasRef.current) {
            const canvasCtx = canvasRef.current.getContext('2d');
            canvasCtx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        }
    };
    
    const setupVisualizer = (stream: MediaStream) => {
        const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
        audioContextRef.current = audioContext;
        const analyser = audioContext.createAnalyser();
        const source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);
        analyser.fftSize = 256;
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        const canvas = canvasRef.current;
        if (!canvas) return;
        const canvasCtx = canvas.getContext('2d');
        const draw = () => {
            animationFrameIdRef.current = requestAnimationFrame(draw);
            if (!analyser || !canvasCtx) return;
            analyser.getByteFrequencyData(dataArray);
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

    // ★★★ 改善点：ファイル処理の中断機能 ★★★
    const processAudioInChunks = async (audioBuffer: AudioBuffer, isRecovery = false) => {
        fileAbortControllerRef.current = new AbortController();
        const signal = fileAbortControllerRef.current.signal;

        setIsProcessingFile(true);
        setLoadingMessage('文字起こしを開始します...');
        setTranscript([]);
        try {
            const duration = audioBuffer.duration;
            const chunkSizeInSeconds = 30;
            const totalChunks = Math.ceil(duration / chunkSizeInSeconds);

            for(let i=0; i < totalChunks; i++) {
                if (signal.aborted) {
                    setModalInfo({ show: true, message: '文字起こし処理を中断しました。'});
                    break;
                }
                setLoadingMessage(`文字起こし中... (${i + 1}/${totalChunks})`);
                const startTime = i * chunkSizeInSeconds;
                const endTime = Math.min(startTime + chunkSizeInSeconds, duration);
                const frameOffset = Math.floor(startTime * audioBuffer.sampleRate);
                const frameCount = Math.floor((endTime - startTime) * audioBuffer.sampleRate);
                if (frameCount <= 0) continue;

                const chunkBuffer = audioContextRef.current!.createBuffer(audioBuffer.numberOfChannels, frameCount, audioBuffer.sampleRate);
                for (let ch = 0; ch < audioBuffer.numberOfChannels; ch++) {
                    chunkBuffer.getChannelData(ch).set(audioBuffer.getChannelData(ch).subarray(frameOffset, frameOffset + frameCount));
                }
                const wavChunkBlob = bufferToWav(chunkBuffer);
                const transcribedText = await transcribeFileRaw(wavChunkBlob, language);
                setTranscript(prev => [...prev, {time: startTime, text: transcribedText}]);
                
                if (i < totalChunks - 1) {
                    await new Promise(resolve => setTimeout(resolve, 1200));
                }
            }
             if (!signal.aborted) {
                setModalInfo({ show: true, message: `${isRecovery ? '復旧データ' : 'ファイル'}の文字起こしが完了しました。`});
            }
        } catch (error: any) {
            console.error("ファイル処理中にエラー:", error);
            setModalInfo({ show: true, message: `文字起こし中にエラーが発生しました: ${error.message}` });
        } finally {
            setIsProcessingFile(false);
            setLoadingMessage('AI準備完了');
            fileAbortControllerRef.current = null;
        }
    };
    
    const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;
        setTranscript([]);
        setSummary('');
        setDownloadLink(URL.createObjectURL(file));
        setActiveTab(0);
        try {
            audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
            setLoadingMessage('音声ファイルを解析中...');
            const arrayBuffer = await file.arrayBuffer();
            const audioBuffer = await audioContextRef.current.decodeAudioData(arrayBuffer);
            await processAudioInChunks(audioBuffer);
        } catch (error) {
            console.error("ファイル読み込みエラー:", error);
            setModalInfo({ show: true, message: 'ファイルの読み込みまたは解析に失敗しました。対応していない形式の可能性があります。'});
        }
    };
    
    const handleCancelFileProcessing = () => {
        if (fileAbortControllerRef.current) {
            fileAbortControllerRef.current.abort();
        }
    };

    const toggleRecording = async () => {
        if (isRecording) {
            setShowStopConfirm(true);
        } else {
            stopVisualizer(true); // マイクチェック用のビジュアライザーを停止
            try {
                await dbManager.clearAudioChunks();
                const constraints = { audio: { deviceId: selectedMicId ? { exact: selectedMicId } : undefined } };
                const stream = await navigator.mediaDevices.getUserMedia(constraints);
                setupVisualizer(stream);
                audioChunksRef.current = [];
                mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
                
                mediaRecorderRef.current.ondataavailable = (e) => {
                    if (e.data.size > 0) {
                        audioChunksRef.current.push(e.data);
                        dbManager.addAudioChunk(e.data);
                    }
                };

                mediaRecorderRef.current.onstop = async () => {
                    stopVisualizer();
                    stream.getTracks().forEach(track => track.stop());
                    
                    const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm;codecs=opus' });
                    if (audioBlob.size === 0) return;

                    setDownloadLink(URL.createObjectURL(audioBlob));
                    
                    try {
                        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
                        const arrayBuffer = await audioBlob.arrayBuffer();
                        const audioBuffer = await audioContextRef.current.decodeAudioData(arrayBuffer);
                        await processAudioInChunks(audioBuffer);
                    } catch(error) {
                        console.error("録音データの処理エラー:", error);
                        setModalInfo({ show: true, message: '録音データの処理中にエラーが発生しました。'});
                    } finally {
                        dbManager.clearAudioChunks();
                    }
                };
                
                setTranscript([]);
                setSummary('');
                setActiveTab(0);
                setIsRecording(true);
                mediaRecorderRef.current.start(1000);

            } catch (err) {
                console.error("マイクアクセス失敗:", err);
                setModalInfo({ show: true, message: 'マイクへのアクセスに失敗しました。'});
            }
        }
    };
    
    const handleConfirmStop = () => {
        if (mediaRecorderRef.current?.state === "recording") {
            mediaRecorderRef.current.stop();
        }
        setIsRecording(false);
        setShowStopConfirm(false);
    };

    const handleProcessRecoveredData = async () => {
        setRecoveryInfo({ show: false, chunkCount: 0 });
        setLoadingMessage('復旧データを処理中...');
        try {
            const recoveredChunks = await dbManager.getAllAudioChunks();
            if (recoveredChunks.length === 0) throw new Error("復旧データが空です。");
            const recoveredBlob = new Blob(recoveredChunks, { type: 'audio/webm;codecs=opus' });
            setDownloadLink(URL.createObjectURL(recoveredBlob));
            audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
            const arrayBuffer = await recoveredBlob.arrayBuffer();
            const audioBuffer = await audioContextRef.current.decodeAudioData(arrayBuffer);
            await processAudioInChunks(audioBuffer, true);
        } catch (error) {
            console.error("復旧データの処理中にエラー:", error);
            setModalInfo({ show: true, message: '復旧データの処理中にエラーが発生しました。'});
        } finally {
            dbManager.clearAudioChunks();
        }
    };
    
    const handleRefineTranscript = async () => {
        if(transcript.length === 0) {
            setModalInfo({ show: true, message: '清書する文字起こしデータがありません。'});
            return;
        }
        setIsRefining(true);
        setLoadingMessage('AIで清書中です...');
        try {
            const rawTranscript = transcript.map(t => t.text).join('\n');
            const refinedText = await refineTranscriptWithMemo(rawTranscript, memoTextRef.current, language);
            const refinedLines = refinedText.split('\n').filter(line => line.trim() !== '');
            const newTranscript = refinedLines.map((line, index) => {
                const originalEntry = transcript[index] || transcript[transcript.length - 1];
                return { time: originalEntry?.time || 0, text: line };
            });
            setTranscript(newTranscript);
            setModalInfo({ show: true, message: 'AIによる清書が完了しました。'});
        } catch (error: any) {
            setModalInfo({ show: true, message: `AIによる清書中にエラーが発生しました: ${error.message}` });
        } finally {
            setIsRefining(false);
            setLoadingMessage('AI準備完了');
        }
    };
    
    const generateSummary = async () => {
        const plainTranscript = transcript.map(item => `${formatTime(item.time)} ${item.text}`).join('\n\n');
        if (!plainTranscript && !memoTextRef.current) {
            setModalInfo({ show: true, message: '要約する文字起こしやメモがありません。'});
            return;
        }
        setIsLoadingAI(true);
        setLoadingMessage('AIが議事録を作成中です...');
        setSummary('');
        setActiveTab(1);
        try {
            const meetingDate = new Date().toLocaleString('ja-JP');
            const prompt = `あなたはプロの議事録作成AIです。提供された以下のタイムスタンプ付き【会議の文字起こし】と【手動メモ】を分析し、その内容から網羅的かつ簡潔な議事録を作成してください。議事録は以下の構造とルールに従って記述してください。---## 議事録### 1. 会議概要**会議日時**: ${meetingDate}会議の目的、主要な議題、および全体的な結論を簡潔にまとめる。### 2. 議論の要点会議で話し合われた重要なポイントや論点を、主要なテーマごとに整理して箇条書きで記述する。### 3. 決定事項会議で合意された事項や結論を明確に箇条書きで記述する。### 4. アクションアイテム (次回以降のタスク)会議で決定された具体的な行動やタスク、担当者、期限を箇条書きで記述する。---**【会議の文字起こし】**${plainTranscript}**【手動メモ】**${memoTextRef.current}`;
            const result = await model.generateContent(prompt);
            setSummary(result.response.text());
        } catch (error: any) {
            setModalInfo({ show: true, message: `議事録の生成中にエラーが発生しました: ${error.message}` });
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
    
    const getButtonState = () => {
        if (isRefining) return { text: 'AIで清書中です...', color: '#28a745', disabled: true };
        if (isProcessingFile) return { text: loadingMessage, color: '#6c757d', disabled: true };
        if (isRecording) return { text: '■ 録音停止', color: '#dc3545', disabled: false };
        if (isLoadingAI) return { text: loadingMessage, color: '#6c757d', disabled: true };
        return { text: '● 録音開始', color: '#007bff', disabled: false };
    };
    const buttonState = getButtonState();

    const customStyles = `
        body { background-color: ${isDarkMode ? '#121212' : '#f4f7f9'}; color: ${isDarkMode ? '#e0e0e0' : '#333'}; transition: background-color 0.3s, color 0.3s; }
        .main-container { background-color: ${isDarkMode ? '#1e1e1e' : '#ffffff'}; padding: 20px 30px; font-family: sans-serif; max-width: 800px; margin: 20px auto; border-radius: 8px; box-shadow: ${isDarkMode ? '0 4px 20px rgba(0,0,0,0.5)' : '0 4px 20px rgba(0,0,0,0.1)'}; transition: background-color 0.3s, box-shadow 0.3s; }
        h1, h2 { color: ${isDarkMode ? '#ffffff' : '#000000'}; }
        .info-box { background-color: ${isDarkMode ? '#2a2a2a' : '#f0f0f0'}; border: 1px solid ${isDarkMode ? '#444' : '#ddd'}; color: ${isDarkMode ? '#e0e0e0' : '#333'}; }
        .react-tabs__tab { background: ${isDarkMode ? '#2a2a2a' : '#f0f0f0'}; border-color: ${isDarkMode ? '#444' : '#ddd'}; color: ${isDarkMode ? '#a0a0a0' : '#333'}; border-bottom: none; }
        .react-tabs__tab--selected { background: ${isDarkMode ? '#1e1e1e' : '#ffffff'}; color: ${isDarkMode ? 'white' : '#007bff'}; border-color: ${isDarkMode ? '#444' : '#ddd'}; border-bottom: 1px solid ${isDarkMode ? '#1e1e1e' : '#ffffff'}; position: relative; top: 1px; }
        .react-tabs__tab-panel--selected { border: 1px solid ${isDarkMode ? '#444' : '#ddd'}; padding: 15px; background-color: ${isDarkMode ? '#2a2a2a' : '#ffffff'}; }
        .transcript-panel { background-color: ${isDarkMode ? '#2a2a2a' : '#fdfdfd'}; border: 1px solid ${isDarkMode ? '#444' : '#ccc'}; white-space: pre-wrap; min-height: 200px; padding: 10px; line-height: 1.8; }
        textarea { background-color: ${isDarkMode ? '#333' : '#fff'}; color: ${isDarkMode ? '#e0e0e0' : '#000'}; border-color: ${isDarkMode ? '#555' : '#ccc'}; }
        .ai-summary-panel { background-color: ${isDarkMode ? '#1e1e1e' : '#d4edda'}; color: ${isDarkMode ? '#e0e0e0' : '#155724'}; padding: 25px; border-radius: 8px; border: 1px solid ${isDarkMode ? '#333' : '#c3e6cb'}; min-height: 250px; white-space: pre-wrap; line-height: 1.7; }
        .ai-summary-panel h2, .ai-summary-panel h3 { color: ${isDarkMode ? '#ffffff' : '#0c5460'}; }
        .ai-summary-panel strong { color: ${isDarkMode ? '#569cd6' : '#004085'}; }
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
        audio { width: 100%; filter: ${isDarkMode ? 'invert(1) contrast(0.8) brightness(1.2)' : 'none'}; }
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
                    <button onClick={toggleRecording} disabled={buttonState.disabled} style={{ fontSize: '16px', padding: '10px 20px', backgroundColor: buttonState.color, color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer', flex: 1 }}>
                        {buttonState.text}
                    </button>
                    <input type="file" ref={fileInputRef} onChange={handleFileChange} accept=".mp3,.m4a,.wav,audio/*" style={{ display: 'none' }} />
                    <button onClick={() => fileInputRef.current?.click()} disabled={isRecording || isLoadingAI || isProcessingFile} style={{ fontSize: '16px', padding: '10px 20px', backgroundColor: '#6c757d', color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer', flex: 1 }}>
                        音声ファイルを読み込む
                    </button>
                </div>
                {isProcessingFile && (
                    <button onClick={handleCancelFileProcessing} style={{ fontSize: '14px', padding: '8px 16px', backgroundColor: '#ffc107', color: 'black', border: 'none', borderRadius: '5px', cursor: 'pointer', marginTop: '10px', width: '100%' }}>
                        文字起こしをキャンセル
                    </button>
                )}
                
                <canvas ref={canvasRef} style={{ width: '100%', height: '60px', marginTop: '15px', borderRadius: '5px' }}></canvas>
                
                <div className="info-box" style={{ marginTop: '15px', padding: '15px', borderRadius: '5px', display: 'flex', gap: '20px' }}>
                     <div style={{flex: 1}}>
                        <label htmlFor="mic-select" style={{display: 'block', marginBottom: '8px'}}><strong>使用するマイクを選択</strong></label>
                        <select id="mic-select" value={selectedMicId} onChange={(e) => handleMicChange(e.target.value)} disabled={isRecording} style={{width: '100%'}}>
                            {devices.length === 0 && <option>マイクが見つかりません</option>}
                            {devices.map(device => ( <option key={device.deviceId} value={device.deviceId}> {device.label || `マイク ${devices.indexOf(device) + 1}`} </option> ))}
                        </select>
                     </div>
                     <div style={{flex: 1}}>
                        <label htmlFor="lang-select" style={{display: 'block', marginBottom: '8px'}}><strong>文字起こし言語</strong></label>
                        <select id="lang-select" value={language} onChange={(e) => setLanguage(e.target.value as any)} disabled={isRecording || isProcessingFile} style={{width: '100%'}}>
                            <option value="auto">自動検出</option>
                            <option value="ja">日本語</option>
                            <option value="en">英語</option>
                        </select>
                     </div>
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
                                <button onClick={handleRefineTranscript} disabled={isLoadingAI || isRefining || transcript.length === 0} style={{fontSize: '14px', padding: '8px 16px', backgroundColor: '#28a745', color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer' }}>
                                    {isRefining ? '清書中です...' : 'AIで清書する'}
                                </button>
                            <button onClick={() => handleDownload(transcript.map(t => `${formatTime(t.time)} ${t.text}`).join('\n\n'), `transcript-${new Date().toISOString().slice(0, 10)}.txt`)} disabled={transcript.length === 0} style={{fontSize: '14px', padding: '8px 16px', backgroundColor: transcript.length === 0 ? '#6c757d' : '#6f42c1', color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer' }}>
                                原文をテキストでダウンロード
                            </button>
                        </div>
                        <div className="transcript-panel">
                           {isLoadingAI || isProcessingFile ? (<p>{loadingMessage}</p>) : 
                               (transcript.length > 0 ? (
                                transcript.map((item, index) => (
                                   <p key={index} style={{margin: '0 0 10px 0'}}>
                                       <span className="timestamp" onClick={() => handleTimestampClick(item.time)}> {formatTime(item.time)} </span>
                                       {item.text}
                                   </p>
                               ))
                           ) : (
                               <p>ここに高精度AIによる文字起こし結果が表示されます...</p>
                           ))}
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
                            {isLoadingAI && summary === '' ? <p>AIが議事録を作成中です...</p> : (
                                summary ? <ReactMarkdown>{summary}</ReactMarkdown> : !isLoadingAI && <p>ここにAIが生成した議事録が表示されます...</p>
                            )}
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
        </>
    );
};

export default App;

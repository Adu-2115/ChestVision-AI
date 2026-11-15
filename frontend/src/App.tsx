import React, { useEffect, useState } from 'react';
import { Result, analyze, errorMessage } from './api';
import Header, { Footer } from './components/Header';
import Landing from './components/Landing';
import Results from './components/Results';
import UploadScreen from './components/UploadScreen';

const DEFAULT_AGE = 60;
const isAnalyzeRoute = () => window.location.hash.replace(/^#\/?/, '').startsWith('analyze');

export default function App() {
  // Two screens, switched by the URL hash: "#/" (landing) and "#/analyze".
  const [onAnalyze, setOnAnalyze] = useState(isAnalyzeRoute);
  useEffect(() => {
    const onHash = () => { setOnAnalyze(isAnalyzeRoute()); window.scrollTo(0, 0); };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [age, setAge] = useState(String(DEFAULT_AGE));
  const [sex, setSex] = useState('Unknown');
  const [result, setResult] = useState<Result | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Release the previous preview URL whenever it is replaced or the app unmounts.
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

  const chooseFile = (f: File) => {
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    window.scrollTo(0, 0);
  };

  const run = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      setResult(await analyze(file, age === '' ? DEFAULT_AGE : Number(age), sex));
      window.scrollTo(0, 0);
    } catch (e) {
      setError(errorMessage(e, 'Something went wrong while analysing the image. Try again.'));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:z-30 focus:m-2 focus:rounded-control focus:bg-card focus:px-3 focus:py-2">
        Skip to content
      </a>
      <Header onAnalyze={onAnalyze} />
      <main id="main" className="flex-1 w-full max-w-[1280px] mx-auto px-4 md:px-6 py-8 md:py-10">
        {!onAnalyze ? (
          <Landing />
        ) : result ? (
          <Results result={result} preview={preview} onReset={reset} />
        ) : (
          <UploadScreen
            file={file} preview={preview} age={age} sex={sex} loading={loading} error={error}
            onFile={chooseFile} onRemove={reset} onAge={setAge} onSex={setSex} onAnalyze={run}
          />
        )}
      </main>
      <Footer />
    </div>
  );
}

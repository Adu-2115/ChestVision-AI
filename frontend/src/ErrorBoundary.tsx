import React from 'react';

interface Props {
  children: React.ReactNode;
}

interface State {
  hasError: boolean;
  errorMessage: string;
}

/**
 * Catches unexpected render errors anywhere in the component tree below it
 * (malformed API responses, unexpected nulls, etc.) and shows a recoverable
 * fallback instead of a blank white screen. Class component is required —
 * error boundaries aren't currently expressible with hooks.
 *
 * Save this as: frontend/src/ErrorBoundary.tsx
 */
export default class ErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, errorMessage: '' };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, errorMessage: error.message || 'Something went wrong.' };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    // In a real production setup this is where you'd send to an error
    // tracking service (Sentry, etc.) — for now just log for visibility.
    console.error('ChestVision AI crashed:', error, info.componentStack);
  }

  handleReload = () => {
    this.setState({ hasError: false, errorMessage: '' });
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-gray-950 text-gray-100 flex items-center justify-center p-6">
          <div className="max-w-md text-center space-y-4">
            <div className="text-5xl">⚠️</div>
            <h1 className="text-xl font-bold text-red-400">Something went wrong</h1>
            <p className="text-gray-400 text-sm">
              ChestVision AI hit an unexpected error and couldn't continue.
              This has been logged. Reloading usually fixes it.
            </p>
            <p className="text-gray-600 text-xs font-mono break-all">
              {this.state.errorMessage}
            </p>
            <button
              onClick={this.handleReload}
              className="mt-2 px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium transition-colors"
            >
              Reload App
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

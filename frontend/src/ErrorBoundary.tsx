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
        <div className="min-h-screen bg-app text-ink flex items-center justify-center p-6">
          <div className="card max-w-md p-6 space-y-3">
            <h1 className="font-serif text-2xl font-semibold">Something went wrong</h1>
            <p className="text-ink-2 leading-relaxed">
              The page hit an unexpected error and couldn't continue. Reloading usually fixes it.
            </p>
            <p className="text-ink-muted text-xs font-mono break-all">
              {this.state.errorMessage}
            </p>
            <button
              onClick={this.handleReload}
              className="min-h-[44px] px-4 rounded-control bg-accent hover:bg-accent-hover text-white font-semibold transition-colors"
            >
              Reload the page
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

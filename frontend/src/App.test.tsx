import React from 'react';
import { render, screen } from '@testing-library/react';
import App from './App';

test('renders ChestVision AI header', () => {
  render(<App />);
  const headerElement = screen.getByText(/ChestVision AI/i);
  expect(headerElement).toBeInTheDocument();
});

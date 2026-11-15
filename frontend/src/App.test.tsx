import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import App from './App';

beforeEach(() => {
  window.localStorage.clear();
  window.location.hash = '';
  window.scrollTo = jest.fn();
});

test('landing page shows the product name and a link to the analyze screen', () => {
  render(<App />);
  expect(screen.getByRole('heading', { level: 1, name: 'ChestVision AI' })).toBeInTheDocument();
  const links = screen.getAllByRole('link', { name: 'Analyze an X-ray' });
  expect(links[0]).toHaveAttribute('href', '#/analyze');
});

test('analyze screen keeps the button disabled until a file is chosen and both boxes are ticked', async () => {
  window.location.hash = '#/analyze';
  render(<App />);
  const analyze = screen.getByRole('button', { name: 'Analyze X-ray' });
  expect(analyze).toBeDisabled();

  const file = new File(['x'], 'xray.png', { type: 'image/png' });
  window.URL.createObjectURL = jest.fn(() => 'blob:preview');
  window.URL.revokeObjectURL = jest.fn();
  fireEvent.change(screen.getByLabelText('Choose a chest X-ray file'), { target: { files: [file] } });
  await screen.findByText('xray.png');   // the file picker handles files asynchronously
  expect(analyze).toBeDisabled();

  screen.getAllByRole('checkbox').forEach(box => fireEvent.click(box));
  expect(analyze).toBeEnabled();
});

test('consent boxes are not shown again once given', () => {
  window.localStorage.setItem('chestvision-consent-v1', 'yes');
  window.location.hash = '#/analyze';
  render(<App />);
  expect(screen.queryAllByRole('checkbox')).toHaveLength(0);
});

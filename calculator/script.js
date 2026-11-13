const display = document.getElementById('display');
const numberButtons = document.querySelectorAll('[data-number]');
const operatorButtons = document.querySelectorAll('[data-operator]');
const equalsButton = document.querySelector('[data-action="equals"]');
const clearButton = document.querySelector('[data-action="clear"]');
const deleteButton = document.querySelector('[data-action="delete"]');
const decimalButton = document.querySelector('[data-action="decimal"]');

let currentOperand = '';
let previousOperand = '';
let operation = null;

function updateDisplay() {
  display.value = currentOperand || '0';
}

function appendNumber(number) {
  if (number === '0' && currentOperand === '0') return;
  if (currentOperand.length >= 12) return;
  currentOperand += number;
}

function appendDecimal() {
  if (!currentOperand.includes('.')) {
    currentOperand += currentOperand === '' ? '0.' : '.';
  }
}

function chooseOperation(op) {
  if (currentOperand === '') return;
  if (previousOperand !== '') compute();
  operation = op;
  previousOperand = currentOperand;
  currentOperand = '';
}

function compute() {
  let result;
  const prev = parseFloat(previousOperand);
  const curr = parseFloat(currentOperand);
  if (isNaN(prev) || isNaN(curr)) return;
  switch (operation) {
    case '+': result = prev + curr; break;
    case '-': result = prev - curr; break;
    case '*': result = prev * curr; break;
    case '/': result = curr !== 0 ? prev / curr : 'Error'; break;
    default: return;
  }
  currentOperand = result === 'Error' ? 'Error' : String(result);
  operation = null;
  previousOperand = '';
}

function clear() {
  currentOperand = '';
  previousOperand = '';
  operation = null;
}

function deleteLast() {
  currentOperand = currentOperand.slice(0, -1);
}

numberButtons.forEach(btn => btn.addEventListener('click', () => {
  appendNumber(btn.dataset.number);
  updateDisplay();
}));

decimalButton.addEventListener('click', () => {
  appendDecimal();
  updateDisplay();
});

operatorButtons.forEach(btn => btn.addEventListener('click', () => {
  chooseOperation(btn.dataset.operator);
  updateDisplay();
}));

equalsButton.addEventListener('click', () => {
  compute();
  updateDisplay();
});

clearButton.addEventListener('click', () => {
  clear();
  updateDisplay();
});

deleteButton.addEventListener('click', () => {
  deleteLast();
  updateDisplay();
});

window.addEventListener('keydown', e => {
  if (e.key >= '0' && e.key <= '9') {
    appendNumber(e.key);
    updateDisplay();
  } else if (e.key === '.') {
    appendDecimal();
    updateDisplay();
  } else if (['+', '-', '*', '/'].includes(e.key)) {
    chooseOperation(e.key);
    updateDisplay();
  } else if (e.key === 'Enter' || e.key === '=') {
    compute();
    updateDisplay();
  } else if (e.key === 'Escape') {
    clear();
    updateDisplay();
  } else if (e.key === 'Backspace') {
    deleteLast();
    updateDisplay();
  }
});

updateDisplay();
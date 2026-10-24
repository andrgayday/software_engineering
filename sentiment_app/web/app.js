const form = document.querySelector('#analysis');
const status = document.querySelector('#status');
const results = document.querySelector('#results');
const labels = {positive: 'Положительная', neutral: 'Нейтральная', negative: 'Отрицательная'};
function card(title, sentiment) {
  const article = document.createElement('article');
  const heading = document.createElement('h2');
  heading.textContent = title;
  const value = document.createElement('p');
  value.className = `sentiment ${sentiment}`;
  value.textContent = labels[sentiment] || sentiment;
  article.append(heading, value);
  return article;
}
form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const text = document.querySelector('#text').value.trim();
  if (!text) { status.textContent = 'Введите текст, содержащий хотя бы один символ.'; return; }
  const button = form.querySelector('button');
  button.disabled = true;
  results.replaceChildren();
  status.textContent = 'Модель анализирует текст…';
  try {
    const response = await fetch('/sentiment', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text})});
    const data = await response.json();
    if (!response.ok) {
      throw new Error(typeof data.detail === 'string' ? data.detail : 'Проверьте текст: от 1 до 5000 символов.');
    }
    results.append(card('Готовая RuBERT-tiny', data.sentiment));
    status.textContent = 'Анализ завершён.';
  } catch (error) {
    status.textContent = error instanceof TypeError ? 'Нет соединения с приложением. Проверьте, запущен ли сервер.' : error.message;
  } finally { button.disabled = false; }
});

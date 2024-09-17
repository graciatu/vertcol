document.querySelector('form').addEventListener('submit', function (e) {
  e.preventDefault();

  const vertebra1 = document.querySelector('#vertebra1').value;
  const vertebra2 = document.querySelector('#vertebra2').value;

  document.querySelector('#results-placeholder').innerHTML = `
    <p>Predicted value based on Vertebra 1: ${vertebra1}</p>
    <p>Predicted value based on Vertebra 2: ${vertebra2}</p>
  `;
});

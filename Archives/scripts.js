
document.getElementById("inputs").addEventListener("submit", setValues);
function checkUnknown(x) {
    if (x === "") {
        return 0;
    }
    else {
        return x;
    }
}
function setValues(){
    event.preventDefault();
}

document.getElementById("inputs").addEventListener("submit", setValues);
function checkUnknown(x) {
    if (x === "") {
        return 0;
    }
    else {
        return x;
    }
}
function setValues(){
    event.preventDefault();


    var c2 = checkUnknown(document.querySelector('#c2').value);
    var c3 = checkUnknown(document.querySelector('#c3').value);
    var c4 = checkUnknown(document.querySelector('#c4').value);
    var c5 = checkUnknown(document.querySelector('#c5').value);
    var c6 = checkUnknown(document.querySelector('#c6').value);
    var c7 = checkUnknown(document.querySelector('#c7').value);
    var t1 = checkUnknown(document.querySelector('#t1').value);
    var t2 = checkUnknown(document.querySelector('#t2').value);
    var t3 = checkUnknown(document.querySelector('#t3').value);
    var t4 = checkUnknown(document.querySelector('#t4').value);
    var t5 = checkUnknown(document.querySelector('#t5').value);
    var t6 = checkUnknown(document.querySelector('#t6').value);
    var t7 = checkUnknown(document.querySelector('#t7').value);
    var t8 = checkUnknown(document.querySelector('#t8').value);
    var t9 = checkUnknown(document.querySelector('#t9').value);
    var t10 = checkUnknown(document.querySelector('#t10').value);
    var t11 = checkUnknown(document.querySelector('#t11').value);
    var t12 = checkUnknown(document.querySelector('#t12').value);
    var l1 = checkUnknown(document.querySelector('#l1').value);
    var l2 = checkUnknown(document.querySelector('#l2').value);
    var l3 = checkUnknown(document.querySelector('#l3').value);
    var l4 = checkUnknown(document.querySelector('#l4').value);
    var l5 = checkUnknown(document.querySelector('#l5').value);


    document.getElementById("results-placeholder").innerHTML = "C2 value is: "+ c2;




    document.getElementById("results-placeholder").innerHTML = "C2 value is: "+ c2;

}

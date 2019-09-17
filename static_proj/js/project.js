// Global Variables

let tester = 0;
let contentToDisplay = '';
document.querySelector('body').addEventListener('click', (e) => {
    if (e.target.classList.contains('hover-link')) {
        console.log(e.target, e.target.classList, e.target.id);
        document.getElementById('link-change').style.display = 'none';
        console.log(e.target.id.split('-')[0])
        console.log(document.getElementById(e.target.id))
        let id=e.target.id
        console.log(document.getElementById(id.split('-')[0]))
        document.getElementById(e.target.id.split('-')[0]).style.display = 'block';
    } else if (e.target.classList.contains('normal-content')) {
        document.getElementById('link-change').style.display = 'block';
        Array.from(document.getElementsByClassName('html-hidden')).forEach(e => {
            e.style.display = "none ";
        })
    }
})
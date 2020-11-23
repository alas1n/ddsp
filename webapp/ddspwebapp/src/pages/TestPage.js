// import React from "react";
// import logo from "./../assests/logo.svg";
// // import "./App.css";

// const TestPage = () => {
//   return (
//     <div className="App">
//       <header className="App-header">
//         <img src={logo} className="App-logo" alt="logo" />
//         <p>
//           Edit <code>src/App.js</code> and save to reload.
//         </p>
//         <a
//           className="App-link"
//           href="https://reactjs.org"
//           target="_blank"
//           rel="noopener noreferrer"
//         >
//           Learn React
//         </a>
//       </header>
//     </div>
//   );
// };

// export default TestPage;
import React, { Component } from "react";

class App extends Component {
  constructor(props) {
    super(props);

    this.state = {
      list: [42, 33, 68],
    };
  }

  onUpdateItem = (i) => {
    this.setState((state) => {
      const list = state.list.map((item, j) => {
        if (j === i) {
          return item + 1;
        } else {
          return item;
        }
      });

      return {
        list,
      };
    });
  };

  render() {
    return (
      <div>
        <ul>
          {this.state.list.map((item, index) => (
            <li key={item}>
              The person is {item} years old.
              <button type="button" onClick={() => this.onUpdateItem(index)}>
                Make me one year older
              </button>
            </li>
          ))}
        </ul>
      </div>
    );
  }
}

export default App;

import React, { Component } from "react";

class App extends Component {
  constructor(props) {
    super(props);

    this.state = {
      list: [42, 33, 68],
    };
  }

  onUpdateItems = () => {
    this.setState((state) => {
      return state.list.map((item) => item + 1);
    });
  };

  render() {
    return (
      <div>
        <ul>
          {this.state.list.map((item) => (
            <li key={item}>The person is {item} years old.</li>
          ))}
        </ul>

        <button type="button" onClick={this.onUpdateItems}>
          Make everyone one year older
        </button>
      </div>
    );
  }
}

export default App;

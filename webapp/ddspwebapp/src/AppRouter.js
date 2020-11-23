import React from "react";
import "./style/main.scss";
import { BrowserRouter as Router, Switch, Route, Link } from "react-router-dom";

// PAGES:
import TestPage from "./pages/TestPage";
import RecorderPage from "./pages/RecorderPage";

const AppRouter = () => {
  return (
    <Router>
      <div>
        <nav>
          <ul>
            <li>
              <Link to="/test">TestPage</Link>
            </li>
            <li>
              <Link to="/recorder">Recorder</Link>
            </li>
          </ul>
        </nav>

        {/* A <Switch> looks through its children <Route>s and
          renders the first one that matches the current URL. */}
        <Switch>
          <Route path="/test">
            <TestPage />
          </Route>
          <Route path="/recorder">
            <RecorderPage />
          </Route>
          <Route path="/">{/* <Home /> */}</Route>
        </Switch>
      </div>
    </Router>
  );
};

export default AppRouter;

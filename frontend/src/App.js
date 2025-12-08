import './App.css';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Home from './pages/Home';
import Detail from './pages/Detail';
import LoadingDetail from './components/LoadingDetail';

function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/detail" element={<Detail />} />
          <Route path="/loading" element={<LoadingDetail />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}

export default App;

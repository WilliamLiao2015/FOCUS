import json

from http.server import BaseHTTPRequestHandler, HTTPServer

from chains import get_chain


chain = get_chain()


class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            return
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        post_data = json.loads(post_data)
        results = chain.invoke(post_data)
        results["log"] = results["log"][:-1]
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(results).encode("utf-8"))


if __name__ == "__main__":
    print("Starting server")
    server = HTTPServer(("localhost", 8080), RequestHandler)
    server.serve_forever()

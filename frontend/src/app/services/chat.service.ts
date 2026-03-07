import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface ChatRequest {
    message: string;
}

export interface ChatResponse {
    response: string;
}

@Injectable({
    providedIn: 'root'
})
export class ChatService {
    private apiUrl = 'http://localhost:8000/chat';

    constructor(private http: HttpClient) { }

    sendMessage(message: string): Observable<ChatResponse> {
        return this.http.post<ChatResponse>(this.apiUrl, { message });
    }
}

import { Component, CUSTOM_ELEMENTS_SCHEMA } from '@angular/core';
import { CommonModule } from '@angular/common';
import { NbChatModule, NbLayoutModule, NbThemeModule } from '@nebular/theme';
import { ChatService } from '../../services/chat.service';

@Component({
    selector: 'app-chat',
    standalone: true,
    imports: [CommonModule, NbChatModule, NbLayoutModule, NbThemeModule],
    schemas: [CUSTOM_ELEMENTS_SCHEMA],
    template: `
    <nb-layout>
      <nb-layout-column>
        <nb-chat title="AI Assistant" size="large" status="primary">
          <nb-chat-message *ngFor="let msg of messages"
            [type]="msg.type"
            [message]="msg.text"
            [reply]="msg.reply"
            [sender]="msg.sender"
            [date]="msg.date"
            [avatar]="msg.avatar">
          </nb-chat-message>

          <nb-chat-form (send)="sendMessage($event)" [dropFiles]="false">
          </nb-chat-form>
        </nb-chat>
      </nb-layout-column>
    </nb-layout>
  `,
    styles: [`
    nb-layout-column {
      display: flex;
      justify-content: center;
      padding: 2rem;
      height: 100vh;
    }
    nb-chat {
      width: 100%;
      max-width: 800px;
      height: 100%;
    }
  `]
})
export class ChatComponent {
    messages: any[] = [
        {
            text: 'Hello! How can I help you today?',
            date: new Date(),
            reply: false,
            sender: 'AI Bot',
            avatar: 'https://i.gifer.com/no.gif',
            type: 'text',
        },
    ];

    constructor(private chatService: ChatService) { }

    sendMessage(event: any) {
        const files = event.files;
        const message = event.message;

        // Add user message to UI
        this.messages.push({
            text: message,
            date: new Date(),
            reply: true,
            sender: 'You',
            avatar: 'https://i.pravatar.cc/150?u=user',
            type: 'text',
        });

        // Call backend
        this.chatService.sendMessage(message).subscribe({
            next: (res) => {
                this.messages.push({
                    text: res.response,
                    date: new Date(),
                    reply: false,
                    sender: 'AI Bot',
                    avatar: 'https://i.gifer.com/no.gif',
                    type: 'text',
                });
            },
            error: (err) => {
                console.error('Error sending message:', err);
                this.messages.push({
                    text: 'Sorry, I am having trouble connecting to the brain. Please try again later.',
                    date: new Date(),
                    reply: false,
                    sender: 'AI Bot',
                    avatar: 'https://i.gifer.com/no.gif',
                    type: 'text',
                });
            }
        });
    }
}

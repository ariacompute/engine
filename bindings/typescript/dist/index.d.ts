export type ChatMessage = {
    role: string;
    content: string;
};
export declare class Engine {
    private lib;
    private handle;
    constructor(bundlePath: string);
    close(): void;
    complete(messages: ChatMessage[], options?: Record<string, unknown>, tools?: unknown[]): any;
    embed(input: string): any;
    transcribe(pcm: Buffer): any;
}
